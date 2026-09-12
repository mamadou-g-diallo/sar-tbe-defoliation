"""
OS1 — carte de sévérité TBE superposée au composite Sentinel-1 (VV_dB), pour
voir concrètement où se situent les zones de défoliation dans l'AOI plutôt
que de s'appuyer uniquement sur des statistiques agrégées (boxplots, SHAP).

Utilisation :
    conda activate hw_senegal
    TBE_AOI=<nom_aoi> python3 os1_severity_map.py
"""

import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from matplotlib.patches import Patch
from rasterio.plot import show as rio_show

from config import AOI_POLYGON, PROC_DIR, RESULT_DIR, SEVERITY_LABELS, YEARS
from os1_join_defoliation import load_tbe_polygons

YEAR = max(YEARS)  # année la plus récente disponible
SEVERITY_COLORS = {1: "#ffd700", 2: "#ff8c00", 3: "#d62728"}  # léger/modéré/grave


def main():
    vv_path = PROC_DIR / f"vv_db_{YEAR}.tif"

    fig, ax = plt.subplots(figsize=(9, 9))
    with rasterio.open(vv_path) as src:
        raster_crs = src.crs
        rio_show(src, ax=ax, cmap="gray", vmin=-25, vmax=-5)

    tbe = load_tbe_polygons()
    year_gdf = tbe[tbe["year"] == YEAR].to_crs(raster_crs)
    print(f"Année {YEAR} : {len(year_gdf)} polygones TBE dans l'AOI")

    for sev, color in SEVERITY_COLORS.items():
        subset = year_gdf[year_gdf["severity"] == sev]
        print(f"  {SEVERITY_LABELS[sev]}: {len(subset)} polygones")
        if len(subset):
            subset.plot(ax=ax, facecolor=color, edgecolor=color, alpha=0.55, linewidth=0.3)

    aoi_proj = gpd.GeoSeries([AOI_POLYGON], crs="EPSG:4326").to_crs(raster_crs)
    aoi_proj.boundary.plot(ax=ax, color="cyan", linewidth=1.5)

    legend_elems = [
        Patch(facecolor=c, edgecolor=c, alpha=0.6, label=SEVERITY_LABELS[s])
        for s, c in SEVERITY_COLORS.items()
    ]
    legend_elems.append(Patch(facecolor="none", edgecolor="cyan", label="Limite AOI"))
    ax.legend(handles=legend_elems, loc="lower right", fontsize=9)
    ax.set_title(
        f"Sévérité de défoliation TBE ({YEAR}) sur composite Sentinel-1 VV (dB)\n"
        "Saguenay–Lac-Saint-Jean — fond noir = hors couverture de l'orbite retenue"
    )
    ax.set_xlabel("Est (m, UTM19N)")
    ax.set_ylabel("Nord (m, UTM19N)")
    fig.tight_layout()
    out = RESULT_DIR / f"severity_map_{YEAR}.png"
    fig.savefig(out, dpi=150)
    print(f"\nCarte -> {out}")


if __name__ == "__main__":
    main()
