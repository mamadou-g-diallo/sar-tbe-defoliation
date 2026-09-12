"""
Descripteurs radar dérivés des composites Sentinel-1 RTC annuels :
rétrodiffusion en dB (VV, VH) et indice de végétation radar dual-pol (RVI),
sensible à la structure/densité du houppier :

    RVI = 4 * VH / (VV + VH)      (calculé en linéaire, converti en dB seulement
                                    pour VV/VH - le RVI reste sans dimension)

Formulation générale du RVI (Kim & van Zyl, 2009), adaptée au dual-pol C-band
et utilisée en suivi de perturbations forestières (ex. Nasirzadehdizaji et
al., 2019). Une défoliation se traduit typiquement par une chute de VH_dB et
du RVI (moins de diffusion de volume dans le houppier) et une remontée
relative de VV_dB (contribution accrue du sol/sous-bois et des branches
dénudées).

Utilisation :
    conda activate hw_senegal
    TBE_AOI=<nom_aoi> python3 os1_feature_extraction.py
"""

import numpy as np
import rasterio

from config import YEARS, RAW_DIR, PROC_DIR, NODATA


def read(path):
    with rasterio.open(path) as ds:
        return ds.read(1), ds.profile


def to_db(linear, valid):
    db = np.full(linear.shape, np.nan, dtype=np.float32)
    ok = valid & (linear > 0)
    db[ok] = 10 * np.log10(linear[ok])
    return db


def save(path, arr, profile, nodata=NODATA):
    prof = dict(profile)
    prof.update(dtype="float32", nodata=nodata, compress="LZW")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(np.where(np.isnan(arr), nodata, arr).astype(np.float32), 1)


MIN_VALID_YEARS = 3  # minimum d'années valides pour calculer une référence pixel fiable


def compute_anomalies(tags=("vv_db", "vh_db", "rvi")):
    """Anomalie temporelle par pixel = valeur de l'année - moyenne pluriannuelle
    au même pixel. Comme toutes les années sont acquises sur la même orbite
    relative (cf. os1_download_data.py), l'angle d'incidence local et les effets
    statiques de terrain/sous-bois sont constants dans le temps pour un pixel
    donné : les soustraire élimine ce bruit géométrique sans avoir besoin
    d'une bande d'angle d'incidence (absente des produits RTC de Planetary
    Computer - vérifié directement sur les assets STAC)."""
    for tag in tags:
        stack, profile = [], None
        for year in YEARS:
            path = PROC_DIR / f"{tag}_{year}.tif"
            if not path.exists():
                continue
            arr, profile = read(path)
            arr = np.where(arr == NODATA, np.nan, arr)
            stack.append(arr)
        if len(stack) < MIN_VALID_YEARS:
            print(f"  [!] {tag}: seulement {len(stack)} année(s) disponible(s), anomalie ignorée")
            continue

        cube = np.stack(stack)  # (n_annees, H, W)
        n_valid_px = np.sum(~np.isnan(cube), axis=0)
        baseline = np.where(n_valid_px >= MIN_VALID_YEARS, np.nanmean(cube, axis=0), np.nan)
        save(PROC_DIR / f"baseline_{tag}.tif", baseline, profile)

        for year in YEARS:
            path = PROC_DIR / f"{tag}_{year}.tif"
            if not path.exists():
                continue
            arr, _ = read(path)
            arr = np.where(arr == NODATA, np.nan, arr)
            anom = arr - baseline
            save(PROC_DIR / f"anom_{tag}_{year}.tif", anom, profile)

        print(f"  {tag}: référence pixel calculée sur {len(stack)} années, "
              f"anomalies écrites pour chaque année")


def main():
    for year in YEARS:
        vv_path = RAW_DIR / f"s1_{year}_vv.tif"
        vh_path = RAW_DIR / f"s1_{year}_vh.tif"
        if not vv_path.exists():
            print(f"  [!] {year}: composite manquant, exécuter os1_download_data.py d'abord")
            continue

        vv, profile = read(vv_path)
        vh, _ = read(vh_path)
        valid = (vv != NODATA) & (vh != NODATA)

        vv_db = to_db(vv, valid)
        vh_db = to_db(vh, valid)

        denom = vv + vh
        rvi = np.full(vv.shape, np.nan, dtype=np.float32)
        ok = valid & (denom > 0)
        rvi[ok] = 4 * vh[ok] / denom[ok]

        save(PROC_DIR / f"vv_db_{year}.tif", vv_db, profile)
        save(PROC_DIR / f"vh_db_{year}.tif", vh_db, profile)
        save(PROC_DIR / f"rvi_{year}.tif", rvi, profile)

        n_valid = int(valid.sum())
        print(f"  {year}: VV_dB [{np.nanmin(vv_db):.1f}, {np.nanmax(vv_db):.1f}] dB, "
              f"RVI moyen {np.nanmean(rvi):.3f}  ({n_valid} px valides)")

    print("\nCalcul des anomalies temporelles par pixel (correction géométrie/orbite)...")
    compute_anomalies()

    print(f"\nTerminé. Rasters dérivés dans : {PROC_DIR}")


if __name__ == "__main__":
    main()
