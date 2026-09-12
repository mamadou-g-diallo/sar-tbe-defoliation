"""
OS2 (suite) — texture GLCM (Haralick) sur VV_dB/VH_dB comme descripteur
d'ordre supérieur à l'amplitude seule, testée directement contre la même
vérité terrain TBE que l'OS1/OS2 (amplitude brute, amplitude corrigée en
anomalie).

Motivation : les tests précédents (os2_water_cloud_model.py,
os2_anomaly_correction.py) montrent que l'amplitude seule (VV/VH, brute ou
corrigée de l'angle d'incidence) ne sépare quasiment pas les classes de
sévérité. Une défoliation change la structure spatiale du houppier
(hétérogénéité, trouées) sans forcément déplacer beaucoup le niveau moyen de
rétrodiffusion - la texture (comment les valeurs de pixels voisins co-varient)
est un candidat direct pour capter cet effet structural.

Par polygone (TBE + témoins, mêmes géométries que os1_join_defoliation.py), sur
chaque bande VV_dB/VH_dB : matrice de co-occurrence de niveaux de gris (GLCM,
distance=1, 4 orientations moyennées), puis contraste, homogénéité, énergie
(ASM) et entropie - calculés directement sur les pixels du polygone plutôt
que sur une fenêtre glissante pleine image (beaucoup plus rapide, et aligné
avec l'unité d'analyse déjà utilisée partout ailleurs dans ce dépôt).

Utilisation :
    conda activate hw_senegal
    TBE_AOI=<nom_aoi> python3 os2_texture_glcm.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from config import YEARS, PROC_DIR, RESULT_DIR, SEVERITY_LABELS
from os1_join_defoliation import load_tbe_polygons, negative_samples

LEVELS = 32
DB_RANGE = {"vv_db": (-42.0, -4.0), "vh_db": (-49.0, -10.0)}  # bornes p1-p99, cf. README
MIN_VALID_PX = 16  # GLCM peu fiable en dessous (~0.64 ha à 20 m)
ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]


def quantize(patch, dmin, dmax):
    """0 réservé aux pixels invalides ; 1..LEVELS pour les valeurs valides."""
    q = np.zeros(patch.shape, dtype=np.uint8)
    valid = ~np.isnan(patch)
    clipped = np.clip(patch[valid], dmin, dmax)
    bins = np.linspace(dmin, dmax, LEVELS + 1)
    q[valid] = np.clip(np.digitize(clipped, bins), 1, LEVELS)
    return q, valid


def glcm_features(patch, dmin, dmax):
    q, valid = quantize(patch, dmin, dmax)
    if valid.sum() < MIN_VALID_PX:
        return None

    com = graycomatrix(q, distances=[1], angles=ANGLES, levels=LEVELS + 1,
                        symmetric=True, normed=False).astype(float)
    com = com[1:, 1:, :, :]  # retire la ligne/colonne du niveau "invalide" (0)
    totals = com.sum(axis=(0, 1), keepdims=True)
    totals[totals == 0] = 1.0
    com_norm = com / totals

    contrast = graycoprops(com_norm, "contrast").mean()
    homogeneity = graycoprops(com_norm, "homogeneity").mean()
    energy = graycoprops(com_norm, "energy").mean()
    p = com_norm[com_norm > 0]
    entropy = float(-(p * np.log2(p)).sum() / com_norm.shape[-1])  # moyenne sur les 4 angles

    return dict(contrast=contrast, homogeneity=homogeneity, energy=energy, entropy=entropy)


def zonal_texture(gdf_year, year):
    rows = []
    rasters = {tag: rasterio.open(PROC_DIR / f"{tag}_{year}.tif") for tag in DB_RANGE}
    try:
        for _, row in gdf_year.iterrows():
            feat = {}
            ok = True
            for tag, (dmin, dmax) in DB_RANGE.items():
                src = rasters[tag]
                geom = [row.geometry]
                try:
                    out, _ = rasterio.mask.mask(src, geom, crop=True, nodata=np.nan)
                except ValueError:
                    ok = False
                    break
                patch = out[0]
                patch = np.where(patch == src.nodata, np.nan, patch)
                props = glcm_features(patch, dmin, dmax)
                if props is None:
                    ok = False
                    break
                for k, v in props.items():
                    feat[f"{tag}_{k}"] = v
            rows.append(feat if ok else None)
    finally:
        for r in rasters.values():
            r.close()
    return rows


def main():
    tbe = load_tbe_polygons()
    tables = []
    for year in YEARS:
        vv_path = PROC_DIR / f"vv_db_{year}.tif"
        if not vv_path.exists():
            continue
        with rasterio.open(vv_path) as ref:
            raster_crs = ref.crs

        year_gdf = tbe[tbe["year"] == year].reset_index(drop=True)
        neg = negative_samples(year, year_gdf.geometry)
        import geopandas as gpd
        combined = gpd.GeoDataFrame(
            pd.concat([year_gdf, neg], ignore_index=True), crs="EPSG:4326"
        ).to_crs(raster_crs)
        if combined.empty:
            continue

        feats = zonal_texture(combined, year)
        out = pd.DataFrame([f for f in feats if f is not None])
        kept_idx = [i for i, f in enumerate(feats) if f is not None]
        out["severity"] = combined["severity"].values[kept_idx]
        out["year"] = year
        tables.append(out)
        print(f"  {year}: {len(combined)} polygones -> {len(out)} avec texture valide "
              f"(>= {MIN_VALID_PX} px)")

    full = pd.concat(tables, ignore_index=True)
    out_csv = RESULT_DIR / "sar_defoliation_samples_texture.csv"
    full.to_csv(out_csv, index=False)
    print(f"\n{len(full)} échantillons -> {out_csv}")
    print(full.groupby("severity").size())

    # --- comparaison au plafond/RF de l'amplitude seule -------------------------
    tex_features = [c for c in full.columns if c not in ("severity", "year")]
    print("\nPlafond théorique (4 classes) par descripteur de texture :")
    for col in tex_features:
        means = full.groupby("severity")[col].transform("mean")
        print(f"  {col}: R² = {r2_score(full[col], means):.4f}")

    X, y = full[tex_features].dropna(), None
    valid_idx = X.index
    y = full.loc[valid_idx, "severity"].astype(int)
    clf = RandomForestClassifier(n_estimators=400, max_depth=6, class_weight="balanced", random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=cv)
    acc = accuracy_score(y, y_pred)
    print(f"\nExactitude Random Forest (texture seule, 5-fold) : {acc:.3f}  (n={len(X)})")
    print("Rappel (os2_anomaly_correction.py) : amplitude brute = 0.383, amplitude anomalie-corrigée = 0.407")

    best_col = max(tex_features, key=lambda c: r2_score(full[c], full.groupby("severity")[c].transform("mean")))
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in sorted(y.unique()):
        ax.hist(X.loc[y == c, best_col], bins=20, alpha=0.5, label=SEVERITY_LABELS[c], density=True)
    ax.set_xlabel(best_col)
    ax.set_ylabel("Densité")
    ax.set_title(f"Descripteur de texture le plus discriminant : {best_col}")
    ax.legend()
    fig.tight_layout()
    out_fig = RESULT_DIR / "os2_texture_glcm.png"
    fig.savefig(out_fig, dpi=150)
    print(f"\nFigure -> {out_fig}")


if __name__ == "__main__":
    main()
