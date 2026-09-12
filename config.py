"""Paramètres du prototype OS1 - signal SAR vs sévérité de défoliation (TBE),
forêt boréale du Québec.

Zone d'étude sélectionnée par la variable d'environnement TBE_AOI (nom d'un
fichier .shp dans data/shp/, sans l'extension), sur le même principe que les
projets sentinel2_kolda / dakar_flood_pikine :

    TBE_AOI=matane python3 os1_download_data.py

Chaque AOI a son propre sous-dossier de données (data/<aoi>/...).
"""

import os
import unicodedata
from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from rasterio.features import geometry_mask
from shapely.ops import transform as shp_transform

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SHP_DIR = DATA_DIR / "shp"
TBE_RAW_DIR = DATA_DIR / "tbe_raw"  # archive MRNF brute (shapefile 2014-aujourd'hui)

# --- Zone d'étude active ----------------------------------------------------------
_AOI_RAW = os.environ.get("TBE_AOI", "saguenay_lsj")
_shp_path = SHP_DIR / f"{_AOI_RAW}.shp"
if _shp_path.exists():
    _gdf = gpd.read_file(_shp_path).to_crs("EPSG:4326")
    AOI_POLYGON = _gdf.union_all()
    AOI_BBOX = tuple(_gdf.total_bounds)
    AOI_AREA_KM2 = float(_gdf.to_crs(_gdf.estimate_utm_crs()).area.sum() / 1e6)
else:
    AOI_POLYGON = AOI_BBOX = AOI_AREA_KM2 = None  # pas encore défini (cf. select_aoi.py)

AOI_NAME = unicodedata.normalize("NFKD", _AOI_RAW.lower().replace(" ", "_")).encode("ascii", "ignore").decode()
AOI_DISPLAY_NAME = _AOI_RAW

# --- Fenêtre d'observation (Sentinel-1, composites estivaux) ----------------------
# Fenêtre de fin d'été : défoliation de l'année déjà exprimée (la repousse de
# feuillage de compensation ne masque pas encore le signal), sol/sous-bois sec
# (moins de bruit lié à l'humidité du sol sous couvert clairsemé).
YEARS = [2017, 2019, 2021, 2023, 2025]
SEASON_START_MD = "07-15"
SEASON_END_MD = "08-31"

RESOLUTION = 20.0  # m - grille de travail (compromis bruit de speckle / taille des peuplements TBE)

# --- Source STAC (Microsoft Planetary Computer, gratuit, sans compte) -------------
STAC_ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1"
S1_COLLECTION = "sentinel-1-rtc"  # gamma0, déjà corrigé du relief

NODATA = -9999.0

# --- Classes de sévérité TBE (relevé aérien MRNF, couche TBE_Defoliation_Annuelle,
# cf. fiche de métadonnées fd_wms_tordeuse_bourgeons_epinette.pdf) -----------------
# Champ ANNEE = année du relevé ; champ Ia = code d'intensité (1/2/3).
# 0 = non affecté, ajouté ici pour l'échantillonnage négatif (zones de l'AOI
# sans aucun polygone TBE cette année-là).
TBE_YEAR_FIELD = "ANNEE"
TBE_SEVERITY_FIELD = "Ia"
SEVERITY_LABELS = {0: "non_affecte", 1: "leger", 2: "modere", 3: "grave"}

# --- Chemins (namespacés par AOI) -------------------------------------------------
AOI_DATA_DIR = DATA_DIR / AOI_NAME
RAW_DIR = AOI_DATA_DIR / "raw"
PROC_DIR = AOI_DATA_DIR / "processed"
RESULT_DIR = AOI_DATA_DIR / "results"
for _d in (RAW_DIR, PROC_DIR, RESULT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def aoi_mask_for_grid(transform, width, height, crs) -> np.ndarray:
    """Masque booléen (True = intérieur de l'AOI) sur une grille raster donnée."""
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    geom_proj = shp_transform(lambda x, y: tr.transform(x, y), AOI_POLYGON)
    return geometry_mask([geom_proj], out_shape=(height, width), transform=transform, invert=True)
