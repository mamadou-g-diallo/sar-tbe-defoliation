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
    TBE_AOI=<nom_aoi> python3 feature_extraction.py
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


def main():
    for year in YEARS:
        vv_path = RAW_DIR / f"s1_{year}_vv.tif"
        vh_path = RAW_DIR / f"s1_{year}_vh.tif"
        if not vv_path.exists():
            print(f"  [!] {year}: composite manquant, exécuter download_data.py d'abord")
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

    print(f"\nTerminé. Rasters dérivés dans : {PROC_DIR}")


if __name__ == "__main__":
    main()
