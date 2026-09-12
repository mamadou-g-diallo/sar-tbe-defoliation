"""
Série temporelle Sentinel-1 RTC (gamma0), composites estivaux médians par
année, sur une même trace orbitale : même orbite relative -> angle
d'incidence quasi constant d'une année à l'autre, condition nécessaire pour
attribuer une variation inter-annuelle de rétrodiffusion à un changement de
couvert plutôt qu'à un changement de géométrie de visée. Source : catalogue
STAC public Microsoft Planetary Computer (gratuit, sans compte ni jeton).

Utilisation :
    conda activate hw_senegal
    TBE_AOI=<nom_aoi> python3 os1_download_data.py
"""

from collections import Counter

import numpy as np
import planetary_computer as pc
import rasterio
from pyproj import Transformer
from pystac_client import Client
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from shapely.geometry import box, shape

from config import (
    AOI_BBOX, YEARS, SEASON_START_MD, SEASON_END_MD, RESOLUTION,
    STAC_ENDPOINT, S1_COLLECTION, NODATA, RAW_DIR,
)

AOI_POLY = box(*AOI_BBOX)


def search_year(collection, year):
    catalog = Client.open(STAC_ENDPOINT)
    dt = f"{year}-{SEASON_START_MD}/{year}-{SEASON_END_MD}"
    items = list(catalog.search(collections=[collection], bbox=AOI_BBOX, datetime=dt).items())
    return [it for it in items if shape(it.geometry).intersects(AOI_POLY)]


def pick_common_orbit(items_by_year):
    """Choisit l'orbite relative couvrant le plus grand nombre d'années - c'est
    elle qui sera utilisée partout, pour garder un angle d'incidence comparable
    d'une année à l'autre (cf. docstring du module)."""
    counts = Counter()
    for items in items_by_year.values():
        orbits = {it.properties.get("sat:relative_orbit") for it in items}
        for o in orbits:
            counts[o] += 1
    orbit, n_years = counts.most_common(1)[0]
    print(f"Orbite relative retenue : {orbit} (disponible {n_years}/{len(items_by_year)} années)")
    return orbit


def target_grid(ref_href, resolution=RESOLUTION):
    with rasterio.open(ref_href) as ds:
        crs = ds.crs
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = tr.transform([AOI_BBOX[0], AOI_BBOX[2]], [AOI_BBOX[1], AOI_BBOX[3]])
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    width = int(np.ceil((xmax - xmin) / resolution))
    height = int(np.ceil((ymax - ymin) / resolution))
    transform = from_origin(xmin, ymax, resolution, resolution)
    return transform, width, height, crs


def fetch_band(href, transform, width, height, crs, resampling=Resampling.bilinear):
    with rasterio.open(href) as src:
        dst = np.full((height, width), NODATA, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1), destination=dst,
            src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
            dst_transform=transform, dst_crs=crs, dst_nodata=NODATA,
            resampling=resampling,
        )
    return dst


def save(path, arr, transform, crs):
    profile = {
        "driver": "GTiff", "dtype": "float32", "count": 1,
        "height": arr.shape[0], "width": arr.shape[1],
        "crs": crs, "transform": transform, "nodata": NODATA, "compress": "LZW",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)


def median_composite(hrefs, transform, width, height, crs):
    """Composite médian multi-dates : réduit le bruit de speckle et lisse les
    variations d'humidité de très court terme (pluie récente) au sein de la
    fenêtre saisonnière, mieux que ne le ferait une scène unique."""
    stack = np.stack([fetch_band(h, transform, width, height, crs) for h in hrefs])
    valid = stack != NODATA
    out = np.full((height, width), NODATA, dtype=np.float32)
    any_valid = valid.any(axis=0)
    masked = np.where(valid, stack, np.nan)
    out[any_valid] = np.nanmedian(masked, axis=0)[any_valid]
    return out


def main():
    print("Recherche des scènes Sentinel-1 RTC par année (Planetary Computer)...")
    items_by_year = {y: search_year(S1_COLLECTION, y) for y in YEARS}
    for y, items in items_by_year.items():
        print(f"  {y}: {len(items)} scène(s) trouvée(s)")
        if not items:
            raise SystemExit(
                f"Aucune scène en {y} sur cette fenêtre/AOI - élargir "
                f"SEASON_START_MD/SEASON_END_MD dans config.py ou vérifier l'AOI."
            )

    orbit = pick_common_orbit(items_by_year)

    ref_item = next(
        it for y in YEARS for it in items_by_year[y]
        if it.properties.get("sat:relative_orbit") == orbit
    )
    transform, width, height, crs = target_grid(pc.sign(ref_item).assets["vv"].href)
    print(f"Grille cible : {width}x{height} px @ {RESOLUTION:.0f} m ({crs})")

    for y in YEARS:
        same_orbit = [it for it in items_by_year[y] if it.properties.get("sat:relative_orbit") == orbit]
        use_items = same_orbit if same_orbit else items_by_year[y]
        if not same_orbit:
            print(f"  [!] {y}: aucune scène sur l'orbite {orbit}, repli sur "
                  f"{len(use_items)} scène(s) d'une autre orbite (à signaler dans les résultats)")
        signed = [pc.sign(it) for it in use_items]
        print(f"  {y}: composite médian sur {len(signed)} scène(s) "
              f"({', '.join(it.id for it in use_items)})")
        for pol in ("vv", "vh"):
            hrefs = [it.assets[pol].href for it in signed]
            comp = median_composite(hrefs, transform, width, height, crs)
            out = RAW_DIR / f"s1_{y}_{pol}.tif"
            save(out, comp, transform, crs)
            valid_pct = (comp != NODATA).mean() * 100
            print(f"    {out.name}  (valides: {valid_pct:.0f}%)")

    print(f"\nTerminé. Données dans : {RAW_DIR}")


if __name__ == "__main__":
    main()
