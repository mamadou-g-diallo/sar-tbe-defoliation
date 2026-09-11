"""
OS2 (premier test) — Water Cloud Model (Attema & Ulaby, 1978) calibré sur les
échantillons SAR x sévérité TBE de l'OS1, pour évaluer si un modèle physique à
faible nombre de paramètres explique mieux la relation rétrodiffusion/sévérité
que la simple lecture d'indice (RVI) de l'OS1.

Modèle (par polarisation p, en puissance linéaire) :

    sigma0_p(V) = A_p * V * cos(theta) * (1 - tau2) + tau2 * sigma0_sol_p
    tau2        = exp(-2 * B_p * V / cos(theta))

où V est un descripteur de végétation (contenu en eau/densité du houppier,
sans unité ici), theta l'angle d'incidence, A_p/B_p des coefficients
empiriques à calibrer, sigma0_sol_p la contribution du sol/sous-bois sous un
houppier totalement atténuant.

N'ayant pas encore de mesure directe de V (LAI, contenu en eau - prévu en
OS2/OS3 via données de terrain), ce prototype estime un V effectif par classe
de sévérité (V0 > V1 > V2 > V3, contrainte de monotonie) conjointement avec
A_p, B_p, sigma0_sol_p, par moindres carrés non linéaires sur les moyennes
zonales VV/VH de l'OS1. C'est un test de plausibilité, pas une inversion
pixel à pixel.

Utilisation :
    conda activate hw_senegal
    TBE_AOI=<nom_aoi> python3 os2_water_cloud_model.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.metrics import r2_score

from config import RESULT_DIR, SEVERITY_LABELS

THETA_DEG = 35.0  # angle d'incidence représentatif (IW mi-fauchée) - pas de
                   # bande d'angle local disponible dans ce prototype, cf. limites
COS_THETA = np.cos(np.radians(THETA_DEG))
SEVERITY_CLASSES = [0, 1, 2, 3]  # non_affecte, leger, modere, grave


def wcm_sigma0(V, A, B, sigma0_sol, cos_theta=COS_THETA):
    tau2 = np.exp(-2 * B * V / cos_theta)
    return A * V * cos_theta * (1 - tau2) + tau2 * sigma0_sol


def unpack_params(x):
    """x = [dV0, dV1, dV2, dV3, A, B, sigma0_sol] avec V_c = somme des dV
    décroissants (>=0) -> garantit V0 > V1 > V2 > V3 >= 0 sans contrainte
    d'inégalité explicite dans le solveur."""
    dV = np.abs(x[:4])
    V_by_class = np.cumsum(dV[::-1])[::-1]  # V3=dV3, V2=dV3+dV2, ... V0=somme totale
    A, B, sigma0_sol = x[4], x[5], x[6]
    return V_by_class, A, B, sigma0_sol


def residuals(x, severity_idx, sigma0_obs):
    V_by_class, A, B, sigma0_sol = unpack_params(x)
    V = V_by_class[severity_idx]
    pred = wcm_sigma0(V, A, B, sigma0_sol)
    return pred - sigma0_obs


def fit_polarization(df, col_db, label):
    sigma0_obs = 10 ** (df[col_db] / 10.0)  # dB -> puissance linéaire
    severity_idx = df["severity"].astype(int).values  # 0..3, indexe V_by_class

    x0 = np.array([0.05, 0.05, 0.05, 0.05, 0.1, 0.1, sigma0_obs.min()])
    result = least_squares(
        residuals, x0, args=(severity_idx, sigma0_obs.values),
        bounds=([0, 0, 0, 0, 1e-4, 1e-4, 0], [1, 1, 1, 1, 5, 5, sigma0_obs.max()]),
    )
    V_by_class, A, B, sigma0_sol = unpack_params(result.x)
    pred = wcm_sigma0(V_by_class[severity_idx], A, B, sigma0_sol)

    pred_db = 10 * np.log10(pred)
    obs_db = df[col_db].values
    r2 = r2_score(obs_db, pred_db)
    rmse = float(np.sqrt(np.mean((pred_db - obs_db) ** 2)))

    print(f"\n=== {label} ===")
    print(f"  A={A:.4f}  B={B:.4f}  sigma0_sol={sigma0_sol:.5f} ({10*np.log10(sigma0_sol):.1f} dB)")
    for c, v in zip(SEVERITY_CLASSES, V_by_class):
        print(f"  V[{SEVERITY_LABELS[c]:>12}] = {v:.4f}")
    print(f"  R² (dB) = {r2:.3f}   RMSE = {rmse:.2f} dB   (n={len(df)})")

    return dict(A=A, B=B, sigma0_sol=sigma0_sol, V_by_class=V_by_class,
                r2=r2, rmse=rmse, obs_db=obs_db, pred_db=pred_db, severity_idx=severity_idx)


def class_mean_ceiling(df, col):
    """R² d'un modèle parfait à 4 valeurs (une par classe) = plafond
    théorique que n'importe quel ajustement WCM à V discret par classe peut
    atteindre. Sert de repère pour juger si l'optimiseur a mal convergé ou si
    le signal est réellement absent dans la bande considérée."""
    means = df.groupby("severity")[col].transform("mean")
    return r2_score(df[col], means)


def main():
    df = pd.read_csv(RESULT_DIR / "sar_defoliation_samples.csv")
    df = df.dropna(subset=["vv_db_mean", "vh_db_mean", "severity"])

    print("Plafond théorique (modèle parfait à 4 valeurs, une par classe) :")
    for col in ("vv_db_mean", "vh_db_mean"):
        print(f"  {col}: R² = {class_mean_ceiling(df, col):.4f}")
    print("  (si proche de 0, aucun ajustement WCM à V discret par classe ne")
    print("   peut faire mieux que 0 - le problème n'est pas l'optimiseur)")

    fits = {
        "VV": fit_polarization(df, "vv_db_mean", "VV"),
        "VH": fit_polarization(df, "vh_db_mean", "VH"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (pol, fit) in zip(axes, fits.items()):
        for c in SEVERITY_CLASSES:
            mask = fit["severity_idx"] == c
            ax.scatter(fit["obs_db"][mask], fit["pred_db"][mask], s=14, alpha=0.6,
                       label=SEVERITY_LABELS[c])
        lims = [min(fit["obs_db"].min(), fit["pred_db"].min()),
                max(fit["obs_db"].max(), fit["pred_db"].max())]
        ax.plot(lims, lims, "k--", linewidth=1)
        ax.set_xlabel(f"{pol}_dB observé")
        ax.set_ylabel(f"{pol}_dB prédit (Water Cloud Model)")
        ax.set_title(f"{pol} : R²={fit['r2']:.2f}, RMSE={fit['rmse']:.2f} dB")
        ax.legend(fontsize=8)
    fig.tight_layout()
    out_fig = RESULT_DIR / "os2_wcm_fit.png"
    fig.savefig(out_fig, dpi=150)
    print(f"\nFigure -> {out_fig}")


if __name__ == "__main__":
    main()
