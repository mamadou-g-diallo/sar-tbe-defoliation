"""
OS2 (suite) — comparaison équitable amplitude vs texture vs combiné, sur
EXACTEMENT le même ensemble de polygones (contrairement à
os2_texture_glcm.py, qui régénérait indépendamment son propre tirage de
témoins "non affecté" - comparaison biaisée par la différence de taille
d'échantillon, comme signalé dans le README).

Pour chaque polygone (TBE + témoins), un seul passage calcule à la fois les
statistiques zonales d'amplitude (VV_dB, VH_dB, RVI - moyenne/écart-type) et
la texture GLCM (contraste, homogénéité, énergie, entropie), avec un
identifiant de ligne stable. La texture est NaN pour les polygones trop
petits (< 16 px valides) ; la comparaison finale se fait uniquement sur le
sous-ensemble où les deux jeux de descripteurs sont disponibles.

Utilisation :
    conda activate hw_senegal
    TBE_AOI=<nom_aoi> python3 os2_combined_features.py
"""

import geopandas as gpd
import pandas as pd
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from config import YEARS, PROC_DIR, RESULT_DIR
from join_defoliation import load_tbe_polygons, negative_samples, zonal as zonal_amplitude
from os2_texture_glcm import zonal_texture

AMP_FEATURES = ["vv_db_mean", "vh_db_mean", "rvi_mean", "vv_db_std", "vh_db_std", "rvi_std"]
TEX_FEATURES = ["vv_db_contrast", "vv_db_homogeneity", "vv_db_energy", "vv_db_entropy",
                 "vh_db_contrast", "vh_db_homogeneity", "vh_db_energy", "vh_db_entropy"]


def build_dataset():
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
        combined = gpd.GeoDataFrame(
            pd.concat([year_gdf, neg], ignore_index=True), crs="EPSG:4326"
        )
        if combined.empty:
            continue
        combined["sample_id"] = [f"{year}_{i}" for i in range(len(combined))]

        amp = zonal_amplitude(combined, year)
        if amp is None:
            continue

        combined_proj = combined.to_crs(raster_crs)
        tex_rows = zonal_texture(combined_proj, year)
        tex = pd.DataFrame([t if t is not None else {} for t in tex_rows])

        row = pd.concat(
            [combined[["sample_id", "year", "severity"]].reset_index(drop=True), amp, tex], axis=1
        )
        tables.append(row)
        print(f"  {year}: {len(combined)} échantillons "
              f"({tex.notna().all(axis=1).sum() if not tex.empty else 0} avec texture valide)")

    return pd.concat(tables, ignore_index=True)


def rf_accuracy(df, features):
    X, y = df[features], df["severity"].astype(int)
    clf = RandomForestClassifier(n_estimators=400, max_depth=6, class_weight="balanced", random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=cv)
    return accuracy_score(y, y_pred)


def main():
    full = build_dataset()
    out_csv = RESULT_DIR / "sar_defoliation_samples_combined.csv"
    full.to_csv(out_csv, index=False)
    print(f"\n{len(full)} échantillons (avant filtrage texture) -> {out_csv}")

    subset = full.dropna(subset=AMP_FEATURES + TEX_FEATURES)
    print(f"\nComparaison équitable sur le même sous-ensemble (n={len(subset)}, "
          f"polygones avec texture ET amplitude valides) :")

    acc_amp = rf_accuracy(subset, AMP_FEATURES)
    acc_tex = rf_accuracy(subset, TEX_FEATURES)
    acc_combo = rf_accuracy(subset, AMP_FEATURES + TEX_FEATURES)

    print(f"  Amplitude seule  ({len(AMP_FEATURES)} descripteurs) : {acc_amp:.3f}")
    print(f"  Texture seule    ({len(TEX_FEATURES)} descripteurs) : {acc_tex:.3f}")
    print(f"  Combiné          ({len(AMP_FEATURES)+len(TEX_FEATURES)} descripteurs) : {acc_combo:.3f}")


if __name__ == "__main__":
    main()
