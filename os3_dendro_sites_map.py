"""
OS3 — carte de localisation des 3 sites de chronologie dendrochronologique
(ITRDB) par rapport à l'AOI Sentinel-1, étiquetée avec le code de site, le
nom et l'espèce (champs utilisés dans os3_dendro_check.py).

Utilisation :
    conda activate hw_senegal
    TBE_AOI=<nom_aoi> python3 os3_dendro_sites_map.py
"""

import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

from config import AOI_POLYGON, RESULT_DIR

# lat, lon, code ITRDB, nom du site, espèce (champ "species" ITRDB)
SITES = [
    (48.88, -71.02, "CANA143", "Lac Onatchiway", "PCMA — épinette noire"),
    (48.08, -70.07, "CANA144", "Mont Valin", "ABBA — sapin baumier (hôte principal TBE)"),
    (49.07, -72.10, "CANA141", "Lac Liberal", "ABBA — sapin baumier"),
]


LABEL_OFFSETS = {
    "CANA141": ((10, 12), "left"),
    "CANA143": ((10, -28), "left"),
    "CANA144": ((-10, 14), "right"),
}


def main():
    fig, ax = plt.subplots(figsize=(9, 6.5))

    aoi = gpd.GeoSeries([AOI_POLYGON], crs="EPSG:4326")
    aoi.plot(ax=ax, facecolor="steelblue", edgecolor="cyan", alpha=0.25, linewidth=1.5)
    ax.annotate("AOI Sentinel-1\n(Saguenay–Lac-Saint-Jean)", xy=(-71.3, 48.83), fontsize=9,
                ha="center", color="steelblue")

    pts = gpd.GeoDataFrame(
        {"code": [s[2] for s in SITES], "nom": [s[3] for s in SITES], "espece": [s[4] for s in SITES]},
        geometry=[Point(s[1], s[0]) for s in SITES], crs="EPSG:4326",
    )
    pts.plot(ax=ax, color="crimson", markersize=90, marker="^", zorder=5)

    for _, row in pts.iterrows():
        offset, ha = LABEL_OFFSETS[row["code"]]
        ax.annotate(
            f"{row['code']} — {row['nom']}\nchamp species : {row['espece']}",
            xy=(row.geometry.x, row.geometry.y), xytext=offset, textcoords="offset points",
            fontsize=8.5, color="black", ha=ha,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="crimson", alpha=0.9),
        )

    minx, miny, maxx, maxy = -73.2, 47.7, -69.3, 49.4
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(
        "Sites de chronologie dendrochronologique (ITRDB, Krause & Morin)\n"
        "vs. AOI Sentinel-1 — champs : code ITRDB, nom du site, espèce (species)"
    )
    ax.set_aspect(1.5, adjustable="box")
    fig.tight_layout()
    out = RESULT_DIR.parent / "dendro_sites_map.png"
    fig.savefig(out, dpi=150)
    print(f"Carte -> {out}")


if __name__ == "__main__":
    main()
