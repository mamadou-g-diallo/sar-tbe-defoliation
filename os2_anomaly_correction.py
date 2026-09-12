"""
OS2 (suite) — la géodatabase RTC de Planetary Computer ne fournit aucune bande
d'angle d'incidence local (vérifié directement sur les assets STAC : vv, vh,
tilejson, preview - rien d'autre). Comme toutes les années sont acquises sur
la même orbite relative (cf. os1_download_data.py), l'angle d'incidence et les
effets statiques de terrain/sous-bois sont constants dans le temps pour un
pixel donné : les retirer via une anomalie temporelle par pixel (valeur -
moyenne pluriannuelle au même pixel, cf. os1_feature_extraction.py ->
compute_anomalies) joue le même rôle qu'une correction géométrique, sans
nécessiter la bande manquante.

Ce script compare, avant/après cette correction :
  1. le plafond théorique R² (meilleur modèle possible à 4 valeurs, une par
     classe de sévérité) sur VV_dB, VH_dB, RVI ;
  2. l'exactitude d'un Random Forest 5-fold sur les mêmes descripteurs.

Utilisation :
    conda activate hw_senegal
    TBE_AOI=<nom_aoi> python3 os2_anomaly_correction.py
    (nécessite d'avoir exécuté os1_feature_extraction.py, os1_join_defoliation.py
    ET os2_join_defoliation_anomaly.py au préalable)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from config import RESULT_DIR

RAW_FEATURES = ["vv_db_mean", "vh_db_mean", "rvi_mean", "vv_db_std", "vh_db_std", "rvi_std"]
ANOM_FEATURES = ["vv_db_anom_mean", "vh_db_anom_mean", "rvi_anom_mean",
                  "vv_db_anom_std", "vh_db_anom_std", "rvi_anom_std"]


def class_mean_ceiling(df, col):
    means = df.groupby("severity")[col].transform("mean")
    return r2_score(df[col], means)


def rf_accuracy(df, features):
    X, y = df[features], df["severity"].astype(int)
    clf = RandomForestClassifier(n_estimators=400, max_depth=6, class_weight="balanced", random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=cv)
    return accuracy_score(y, y_pred)


def main():
    raw = pd.read_csv(RESULT_DIR / "sar_defoliation_samples.csv").dropna(subset=RAW_FEATURES + ["severity"])
    anom = pd.read_csv(RESULT_DIR / "sar_defoliation_samples_anomaly.csv").dropna(subset=ANOM_FEATURES + ["severity"])

    print(f"Bruts    : n={len(raw)}")
    print(f"Anomalie : n={len(anom)}\n")

    rows = []
    for label, raw_col, anom_col in [("VV_dB", "vv_db_mean", "vv_db_anom_mean"),
                                       ("VH_dB", "vh_db_mean", "vh_db_anom_mean"),
                                       ("RVI", "rvi_mean", "rvi_anom_mean")]:
        std_raw = raw.groupby("severity")[raw_col].std().mean()
        std_anom = anom.groupby("severity")[anom_col].std().mean()
        r2_raw = class_mean_ceiling(raw, raw_col)
        r2_anom = class_mean_ceiling(anom, anom_col)
        print(f"{label:6s}  écart-type intra-classe moyen : {std_raw:.3f} (brut) -> {std_anom:.3f} (anomalie)  "
              f"[{'réduction' if std_anom < std_raw else 'hausse'} de {abs(1-std_anom/std_raw)*100:.0f}%]")
        print(f"        R² plafond (4 classes)         : {r2_raw:.4f} (brut) -> {r2_anom:.4f} (anomalie)")
        rows.append((label, r2_raw, r2_anom))

    acc_raw = rf_accuracy(raw, RAW_FEATURES)
    acc_anom = rf_accuracy(anom, ANOM_FEATURES)
    print(f"\nExactitude Random Forest (5-fold) : {acc_raw:.3f} (brut) -> {acc_anom:.3f} (anomalie)")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    labels = [r[0] for r in rows] + ["RF accuracy"]
    raw_vals = [r[1] for r in rows] + [acc_raw]
    anom_vals = [r[2] for r in rows] + [acc_anom]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, raw_vals, w, label="Brut")
    ax.bar(x + w / 2, anom_vals, w, label="Anomalie (corrigé)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylabel("R² (plafond 4 classes)  /  exactitude RF")
    ax.set_title("Effet de la correction d'anomalie temporelle par pixel")
    ax.legend()
    fig.tight_layout()
    out_fig = RESULT_DIR / "os2_anomaly_comparison.png"
    fig.savefig(out_fig, dpi=150)
    print(f"\nFigure -> {out_fig}")


if __name__ == "__main__":
    main()
