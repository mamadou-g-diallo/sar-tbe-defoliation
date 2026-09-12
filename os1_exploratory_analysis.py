"""
Exploration Random Forest + SHAP : quelle contribution du VV, VH et du RVI
(et de leur variabilité intra-polygone) à la sévérité de défoliation TBE
observée la même année ? Étape exploratoire de l'OS1 - identifier quels
descripteurs radar portent le signal, avant tout travail de modélisation
physique (Water Cloud Model / MIMICS) ou d'inversion.

Utilisation :
    conda activate hw_senegal
    TBE_AOI=<nom_aoi> python3 os1_exploratory_analysis.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from config import RESULT_DIR, SEVERITY_LABELS

FEATURES = ["vv_db_mean", "vh_db_mean", "rvi_mean", "vv_db_std", "vh_db_std", "rvi_std"]


def main():
    df = pd.read_csv(RESULT_DIR / "sar_defoliation_samples.csv")
    X = df[FEATURES]
    y = df["severity"].astype(int)
    classes = sorted(y.unique())
    class_names = [SEVERITY_LABELS[c] for c in classes]

    print("Distribution des classes :")
    print(y.map(SEVERITY_LABELS).value_counts())

    clf = RandomForestClassifier(n_estimators=400, max_depth=6, class_weight="balanced", random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=cv)
    print("\nValidation croisée (5 plis, hors échantillon) :")
    print(classification_report(y, y_pred, labels=classes, target_names=class_names))

    clf.fit(X, y)

    # --- Figure 1 : RVI moyen par classe de sévérité ---------------------------
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    df.boxplot(column="rvi_mean", by="severity", ax=ax1)
    ax1.set_xticklabels([SEVERITY_LABELS[int(t.get_text())] for t in ax1.get_xticklabels()])
    ax1.set_title("RVI moyen par classe de sévérité TBE")
    ax1.set_xlabel("")
    ax1.set_ylabel("RVI (composite estival)")
    plt.suptitle("")
    fig1.tight_layout()
    fig1.savefig(RESULT_DIR / "rvi_by_severity.png", dpi=150)
    print(f"\nFigure -> {RESULT_DIR / 'rvi_by_severity.png'}")

    # --- Figure 2 : importance des descripteurs (SHAP) --------------------------
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)

    fig2 = plt.figure(figsize=(7, 5))
    shap.summary_plot(shap_values, X, class_names=class_names, plot_type="bar", show=False)
    plt.title("Importance des descripteurs radar (SHAP, moyenne |valeur| par classe)")
    plt.tight_layout()
    fig2.savefig(RESULT_DIR / "shap_summary.png", dpi=150)
    print(f"Figure -> {RESULT_DIR / 'shap_summary.png'}")

    print("\nTerminé.")


if __name__ == "__main__":
    main()
