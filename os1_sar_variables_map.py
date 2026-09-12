"""
OS1 — carte des trois descripteurs radar (VV_dB, VH_dB, RVI) côte à côte sur
l'AOI, pour une année donnée. Complète severity_map_2025.png (qui montre le
VV seul en fond) en donnant une vraie carte de chaque variable utilisée dans
les analyses.

Utilisation :
    conda activate hw_senegal
    TBE_AOI=<nom_aoi> python3 os1_sar_variables_map.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.plot import show as rio_show

from config import AOI_POLYGON, PROC_DIR, RESULT_DIR, YEARS
import geopandas as gpd

YEAR = max(YEARS)

PANELS = [
    ("vv_db", "VV (dB)", "gray", -25, -5),
    ("vh_db", "VH (dB)", "gray", -30, -10),
    ("rvi", "RVI (sans unité)", "YlGn", 0.5, 1.2),
]


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    aoi_series = gpd.GeoSeries([AOI_POLYGON], crs="EPSG:4326")

    for ax, (tag, label, cmap, vmin, vmax) in zip(axes, PANELS):
        path = PROC_DIR / f"{tag}_{YEAR}.tif"
        with rasterio.open(path) as src:
            im = rio_show(src, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax)
            aoi_proj = aoi_series.to_crs(src.crs)
            aoi_proj.boundary.plot(ax=ax, color="cyan", linewidth=1.2)
        ax.set_title(f"{label}\nchamp raster : {tag}_{YEAR}.tif", fontsize=11)
        ax.set_xlabel("Est (m, UTM19N)")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label=label)

    axes[0].set_ylabel("Nord (m, UTM19N)")
    fig.suptitle(
        f"Descripteurs radar Sentinel-1 — composite estival {YEAR}, Saguenay–Lac-Saint-Jean\n"
        "(limite AOI en cyan ; RVI = 4·VH/(VV+VH) en linéaire)",
        fontsize=12,
    )
    fig.tight_layout()
    out = RESULT_DIR / f"sar_variables_map_{YEAR}.png"
    fig.savefig(out, dpi=150)
    print(f"Carte -> {out}")


if __name__ == "__main__":
    main()
