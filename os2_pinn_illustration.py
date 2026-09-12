"""
OS2 (illustration pédagogique, PAS un résultat sur nos données réelles) —
un réseau de neurones informé par la physique (PINN) pour inverser le
potentiel hydrique de l'arbre (Psi, MPa) à partir de la rétrodiffusion SAR,
en s'appuyant sur très peu de mesures de terrain étiquetées.

Pourquoi une illustration synthétique et pas un vrai test sur l'AOI :
os2_water_cloud_model.py a montré que le Water Cloud Model, calibré
librement sur nos données réelles, n'explique presque aucune variance
(R² plafond ≈ 0,003) - parce que les vraies variables d'état physiques
(contenu en eau du houppier mesuré, angle d'incidence local) nous manquent,
pas parce que le modèle ou l'optimiseur sont mauvais. Un PINN ne peut pas
injecter une information physique absente des données ; il n'a donc rien à
démontrer sur l'AOI actuel. Ce script illustre plutôt CE QUE l'approche PINN
apporterait une fois de vraies mesures hydrauliques disponibles (OS2/OS3) :
apprendre à inverser Psi à partir de très peu d'exemples étiquetés, en
s'appuyant sur un décodeur physique différentiable plutôt que sur la seule
supervision.

Chaîne physique jouet (coefficients illustratifs, PAS calibrés sur des
mesures réelles) :

    Psi (potentiel hydrique, MPa) --[courbe pression-volume logistique]-->
    M (teneur en eau du houppier, 0-1) --[≈ descripteur V du WCM]-->
    Water Cloud Model (Attema & Ulaby 1978, mêmes équations que
    os2_water_cloud_model.py) --> sigma0_VV, sigma0_VH

Protocole :
  1. 500 échantillons synthétiques (Psi tiré uniformément, bruit de mesure
     réaliste ajouté sur sigma0 en dB).
  2. Seuls 15 échantillons "connaissent" Psi (simule une petite campagne de
     terrain - le genre de volume réaliste pour une première année d'OS3).
  3. Modèle A (baseline) : régression supervisée pure, entraînée sur les 15
     seuls exemples étiquetés.
  4. Modèle B (PINN) : même réseau, mais un décodeur physique différentiable
     (la chaîne ci-dessus) ajoute un terme de cohérence physique calculable
     sur les 500 échantillons SANS connaître leur Psi (auto-supervision),
     en plus du terme supervisé sur les 15 exemples étiquetés.
  5. Comparaison de l'erreur de reconstruction de Psi sur un jeu de test.

Utilisation :
    conda activate hw_senegal
    python3 os2_pinn_illustration.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from config import RESULT_DIR

torch.manual_seed(0)
np.random.seed(0)

N_TOTAL = 500
N_LABELED = 15
N_TEST = 150
THETA = torch.tensor(np.radians(35.0))
COS_THETA = torch.cos(THETA)

# Coefficients WCM "vrais" (illustratifs - inconnus du PINN, qui doit les
# apprendre implicitement via le décodeur, cf. TrueParams vs PhysicsDecoder)
TRUE_PARAMS = dict(
    A_vv=0.09, B_vv=0.35, s0soil_vv=0.09,
    A_vh=0.05, B_vh=0.55, s0soil_vh=0.025,
)


def psi_to_M(psi, k=2.5, psi0=-1.4, m_max=0.75):
    """Courbe pression-volume logistique jouet : teneur en eau du houppier
    en fonction du potentiel hydrique (plus Psi est négatif -> plus le
    houppier est sec)."""
    return m_max / (1 + torch.exp(-k * (psi - psi0)))


def wcm_sigma0(V, A, B, s0soil):
    tau2 = torch.exp(-2 * B * V / COS_THETA)
    return A * V * COS_THETA * (1 - tau2) + tau2 * s0soil


def forward_chain(psi, params):
    M = psi_to_M(psi)
    vv = wcm_sigma0(M, params["A_vv"], params["B_vv"], params["s0soil_vv"])
    vh = wcm_sigma0(M, params["A_vh"], params["B_vh"], params["s0soil_vh"])
    return vv, vh


def to_db(x):
    return 10 * torch.log10(torch.clamp(x, min=1e-6))


def from_db(x_db):
    return 10 ** (x_db / 10.0)


def make_synthetic_data():
    # Bornes resserrées sur la partie bien identifiable de la courbe pression-volume
    # (psi0=-1.4, k=2.5) : au-delà, la logistique sature et plusieurs Psi produisent
    # quasiment le même M -> inversion mal posée par construction, pas un défaut du PINN.
    psi_true = torch.empty(N_TOTAL).uniform_(-2.2, -0.6)
    vv_lin, vh_lin = forward_chain(psi_true, TRUE_PARAMS)
    vv_db = to_db(vv_lin) + torch.randn(N_TOTAL) * 1.2  # bruit ~1.2 dB, plausible pour un composite
    vh_db = to_db(vh_lin) + torch.randn(N_TOTAL) * 1.2
    X = torch.stack([vv_db, vh_db], dim=1)
    return X, psi_true


class Encoder(nn.Module):
    """SAR (VV_dB, VH_dB) -> Psi. Même architecture pour les deux modèles,
    seule la fonction de perte diffère."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PhysicsDecoder(nn.Module):
    """Chaîne physique différentiable, avec ses propres paramètres A/B/sigma0_soil
    à estimer (initialisés loin des vraies valeurs - le PINN doit aussi les
    apprendre, pas seulement Psi)."""

    def __init__(self):
        super().__init__()
        init = lambda v: nn.Parameter(torch.tensor(float(v)))
        self.A_vv, self.B_vv, self.s0soil_vv = init(0.15), init(0.6), init(0.05)
        self.A_vh, self.B_vh, self.s0soil_vh = init(0.10), init(0.6), init(0.05)

    def forward(self, psi_pred):
        M = psi_to_M(psi_pred)
        vv = wcm_sigma0(M, self.A_vv.abs(), self.B_vv.abs(), self.s0soil_vv.abs())
        vh = wcm_sigma0(M, self.A_vh.abs(), self.B_vh.abs(), self.s0soil_vh.abs())
        return to_db(vv), to_db(vh)


def train_baseline(X_lab, psi_lab, epochs=2000):
    model = Encoder()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(X_lab)
        loss = nn.functional.mse_loss(pred, psi_lab)
        loss.backward()
        opt.step()
    return model


def train_pinn(X_all, X_lab, psi_lab, epochs=2000, lambda_phys=1.0, lambda_data=5.0):
    encoder = Encoder()
    decoder = PhysicsDecoder()
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        psi_pred_all = encoder(X_all)
        vv_rec, vh_rec = decoder(psi_pred_all)
        loss_phys = nn.functional.mse_loss(vv_rec, X_all[:, 0]) + nn.functional.mse_loss(vh_rec, X_all[:, 1])

        psi_pred_lab = encoder(X_lab)
        loss_data = nn.functional.mse_loss(psi_pred_lab, psi_lab)

        loss = lambda_phys * loss_phys + lambda_data * loss_data
        loss.backward()
        opt.step()
    return encoder


def main():
    X, psi_true = make_synthetic_data()
    idx = torch.randperm(N_TOTAL)
    lab_idx, test_idx = idx[:N_LABELED], idx[N_LABELED:N_LABELED + N_TEST]

    X_lab, psi_lab = X[lab_idx], psi_true[lab_idx]
    X_test, psi_test = X[test_idx], psi_true[test_idx]

    print(f"{N_TOTAL} échantillons synthétiques, {N_LABELED} étiquetés (Psi connu), "
          f"{N_TEST} en test.")

    baseline = train_baseline(X_lab, psi_lab)
    pinn = train_pinn(X, X_lab, psi_lab)

    with torch.no_grad():
        pred_baseline = baseline(X_test)
        pred_pinn = pinn(X_test)

    rmse_baseline = torch.sqrt(nn.functional.mse_loss(pred_baseline, psi_test)).item()
    rmse_pinn = torch.sqrt(nn.functional.mse_loss(pred_pinn, psi_test)).item()

    print(f"\nRMSE Psi (MPa) sur le jeu de test (n={N_TEST}), {N_LABELED} exemples étiquetés seulement :")
    print(f"  Régression supervisée seule : {rmse_baseline:.3f} MPa")
    print(f"  PINN (+ cohérence physique) : {rmse_pinn:.3f} MPa")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    for ax, pred, rmse, title in [
        (axes[0], pred_baseline, rmse_baseline, "Régression supervisée seule\n(15 exemples)"),
        (axes[1], pred_pinn, rmse_pinn, "PINN : supervision + cohérence physique\n(15 exemples + 500 non étiquetés)"),
    ]:
        ax.scatter(psi_test.numpy(), pred.numpy(), s=12, alpha=0.5)
        lims = [-2.3, -0.5]
        ax.plot(lims, lims, "k--", linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Psi vrai (MPa)")
        ax.set_title(f"{title}\nRMSE = {rmse:.3f} MPa")
    axes[0].set_ylabel("Psi prédit (MPa)")
    fig.suptitle("Illustration pédagogique (données synthétiques) — PAS un résultat sur l'AOI réel", fontsize=10)
    fig.tight_layout()
    out_fig = RESULT_DIR.parent.parent / "os2_pinn_illustration.png"
    fig.savefig(out_fig, dpi=150)
    print(f"\nFigure -> {out_fig}")


if __name__ == "__main__":
    main()
