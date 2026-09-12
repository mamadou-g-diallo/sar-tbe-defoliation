"""
Variante de join_defoliation.py utilisant les rasters d'ANOMALIE temporelle
par pixel (anom_vv_db_<year>.tif, etc., produits par feature_extraction.py)
plutôt que les valeurs brutes - pour tester si retirer l'effet géométrique
constant par pixel (angle d'incidence local, orbite fixe) améliore la
séparation par sévérité observée dans os2_water_cloud_model.py.

Utilisation :
    conda activate hw_senegal
    TBE_AOI=<nom_aoi> python3 join_defoliation_anomaly.py
"""

import geopandas as gpd
import pandas as pd
import rasterio
from rasterstats import zonal_stats

from config import YEARS, PROC_DIR, RESULT_DIR
from join_defoliation import load_tbe_polygons, negative_samples


def zonal_anomaly(gdf_year, year):
    feats = {}
    for tag in ("vv_db", "vh_db", "rvi"):
        path = PROC_DIR / f"anom_{tag}_{year}.tif"
        if not path.exists():
            print(f"  [!] {path.name} manquant - exécuter feature_extraction.py d'abord")
            return None
        with rasterio.open(path) as src:
            nodata = src.nodata
            raster_crs = src.crs
        gdf_proj = gdf_year.to_crs(raster_crs)
        stats = zonal_stats(gdf_proj, path, stats=["mean", "std"], nodata=nodata, geojson_out=False)
        feats[f"{tag}_anom_mean"] = [s["mean"] for s in stats]
        feats[f"{tag}_anom_std"] = [s["std"] for s in stats]
    return pd.DataFrame(feats)


def main():
    tbe = load_tbe_polygons()
    print(f"{len(tbe)} polygones TBE dans l'AOI, toutes années confondues")

    tables = []
    for year in YEARS:
        year_gdf = tbe[tbe["year"] == year].reset_index(drop=True)
        neg = negative_samples(year, year_gdf.geometry)
        combined = gpd.GeoDataFrame(
            pd.concat([year_gdf, neg], ignore_index=True), crs="EPSG:4326"
        )
        if combined.empty:
            continue
        feats = zonal_anomaly(combined, year)
        if feats is None:
            continue
        out = pd.concat([combined[["year", "severity"]].reset_index(drop=True), feats], axis=1).dropna()
        tables.append(out)
        print(f"  {year}: {len(year_gdf)} polygones TBE + {len(neg)} témoins -> {len(out)} échantillons valides")

    full = pd.concat(tables, ignore_index=True)
    out_csv = RESULT_DIR / "sar_defoliation_samples_anomaly.csv"
    full.to_csv(out_csv, index=False)
    print(f"\n{len(full)} échantillons au total -> {out_csv}")
    print(full.groupby("severity").size())


if __name__ == "__main__":
    main()
