"""
Jointure spatiale : pour chaque année de config.YEARS, croise les polygones
de sévérité TBE (relevé aérien MRNF, champ ANNEE + Ia/NIVEAU) avec les
descripteurs radar (VV_dB, VH_dB, RVI) de la même année, par statistique
zonale (moyenne, écart-type) sur chaque polygone. Ajoute des témoins "non
affecté" (grille régulière dans l'AOI, hors tout polygone TBE de l'année)
pour donner à l'analyse une classe de référence.

Nécessite l'archive locale TBE_Donnees_2014-aujourdhui.zip extraite sous
data/tbe_raw/ (cf. README) - le champ ANNEE + Ia sert directement de vérité
terrain, sans autre traitement.

Utilisation :
    conda activate hw_senegal
    TBE_AOI=<nom_aoi> python3 os1_join_defoliation.py
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats
from shapely.geometry import box

from config import (
    YEARS, AOI_POLYGON, AOI_BBOX, PROC_DIR, RESULT_DIR, TBE_RAW_DIR,
    TBE_YEAR_FIELD, TBE_SEVERITY_FIELD,
)

SEVERITY_TEXT_TO_CODE = {"Leger": 1, "Léger": 1, "Modere": 2, "Modéré": 2, "Grave": 3}


def load_tbe_polygons():
    """L'archive MRNF 2014-aujourd'hui est en réalité une File Geodatabase Esri
    (malgré l'étiquette SHP sur Données Québec) : une seule couche, un polygone
    par (année, secteur défolié)."""
    shp_candidates = list(TBE_RAW_DIR.rglob("*.shp"))
    gdb_candidates = list(TBE_RAW_DIR.rglob("*.gdb"))

    if shp_candidates:
        src, layer = shp_candidates[0], None
    elif gdb_candidates:
        src = gdb_candidates[0]
        layer = gpd.list_layers(src).iloc[0]["name"]
    else:
        raise SystemExit(f"Aucun .shp ni .gdb trouvé sous {TBE_RAW_DIR} - extraire l'archive TBE d'abord.")

    print(f"Lecture de {src.name} (couche={layer}, filtre bbox = AOI)...")
    gdf = gpd.read_file(src, layer=layer, bbox=AOI_BBOX)
    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.intersects(AOI_POLYGON)].copy()

    if TBE_SEVERITY_FIELD in gdf.columns:
        gdf["severity"] = gdf[TBE_SEVERITY_FIELD].astype(int)
    elif "NIVEAU" in gdf.columns:
        gdf["severity"] = gdf["NIVEAU"].map(SEVERITY_TEXT_TO_CODE)
    else:
        raise SystemExit(f"Ni {TBE_SEVERITY_FIELD} ni NIVEAU dans les colonnes : {list(gdf.columns)}")

    gdf["year"] = gdf[TBE_YEAR_FIELD].astype(int)
    return gdf.dropna(subset=["severity"])[["year", "severity", "geometry"]]


def negative_samples(year, affected_geoms, n=60, cell_km=1.5):
    """Grille régulière sur l'AOI, cellules ne touchant aucun polygone TBE de
    l'année -> témoin 'non affecté' (severity=0)."""
    minx, miny, maxx, maxy = AOI_BBOX
    deg = cell_km / 111.0  # approximation degré -> km à cette latitude
    xs = np.arange(minx, maxx, deg)
    ys = np.arange(miny, maxy, deg)
    cells = [box(x, y, x + deg, y + deg) for x in xs for y in ys]
    affected_union = affected_geoms.union_all() if len(affected_geoms) else None
    free = [c for c in cells if c.intersects(AOI_POLYGON) and
            (affected_union is None or not c.intersects(affected_union))]
    rng = np.random.default_rng(42)
    idx = rng.choice(len(free), size=min(n, len(free)), replace=False) if free else np.array([], dtype=int)
    rows = [{"year": year, "severity": 0, "geometry": free[i]} for i in idx]
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def zonal(gdf_year, year):
    feats = {}
    for tag in ("vv_db", "vh_db", "rvi"):
        path = PROC_DIR / f"{tag}_{year}.tif"
        if not path.exists():
            print(f"  [!] {path.name} manquant - exécuter os1_feature_extraction.py d'abord")
            return None
        with rasterio.open(path) as src:
            nodata = src.nodata
            raster_crs = src.crs
        # rasterstats suppose le vecteur déjà dans le CRS du raster - reprojection
        # indispensable ici (polygones TBE en EPSG:4326, rasters en UTM).
        gdf_proj = gdf_year.to_crs(raster_crs)
        stats = zonal_stats(gdf_proj, path, stats=["mean", "std"], nodata=nodata, geojson_out=False)
        feats[f"{tag}_mean"] = [s["mean"] for s in stats]
        feats[f"{tag}_std"] = [s["std"] for s in stats]
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
            print(f"  {year}: aucun échantillon, ignoré")
            continue
        feats = zonal(combined, year)
        if feats is None:
            continue
        out = pd.concat([combined[["year", "severity"]].reset_index(drop=True), feats], axis=1).dropna()
        tables.append(out)
        print(f"  {year}: {len(year_gdf)} polygones TBE + {len(neg)} témoins -> {len(out)} échantillons valides")

    if not tables:
        raise SystemExit("Aucune donnée jointe - vérifier que os1_feature_extraction.py a bien tourné pour ces années.")

    full = pd.concat(tables, ignore_index=True)
    out_csv = RESULT_DIR / "sar_defoliation_samples.csv"
    full.to_csv(out_csv, index=False)
    print(f"\n{len(full)} échantillons au total -> {out_csv}")
    print(full.groupby("severity").size())


if __name__ == "__main__":
    main()
