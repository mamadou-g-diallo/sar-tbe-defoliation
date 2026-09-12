"""
OS4 (test anticipé) — un modèle fondation pré-entraîné sur Sentinel-1 apporte-
t-il plus de signal que les descripteurs manuels (amplitude OS1/OS2, texture
GLCM OS2) ? Test en "linear probing" : le réseau pré-entraîné (SSL4EO-S12,
ResNet50, auto-supervision MoCo sur des paires VV/VH mondiales - Wang et al.,
2022, https://arxiv.org/abs/2211.07044) reste gelé, on extrait juste
l'embedding (2048-d) de chaque polygone et on entraîne un Random Forest
dessus - même protocole que os2_combined_features.py, pour rester comparable.

Par polygone : patch fixe (64x64 px, 20 m -> 1,28 km de côté) centré sur le
centroïde, extrait de vv_db/vh_db (déjà à l'échelle dB attendue par le
prétraitement du modèle : mean=[-12.59,-20.26], std=[5.26,5.91]). Beaucoup de
polygones TBE sont plus petits que ce patch - le patch déborde alors sur le
voisinage immédiat, ce qui est assumé (contexte spatial plutôt que
découpage strict au polygone, cohérent avec la façon dont ces modèles sont
pré-entraînés sur des tuiles régulières, pas des polygones).

Utilisation :
    conda activate hw_senegal
    TBE_AOI=<nom_aoi> python3 os4_foundation_model_probe.py
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from torchgeo.models import ResNet50_Weights, resnet50

from config import YEARS, PROC_DIR, RESULT_DIR
from os1_join_defoliation import load_tbe_polygons, negative_samples

PATCH_PX = 64
WEIGHTS = ResNet50_Weights.SENTINEL1_ALL_MOCO


def load_model():
    model = resnet50(weights=WEIGHTS)
    model.fc = torch.nn.Identity()  # embedding brut (2048-d), pas de tête de classification
    model.eval()
    return model, WEIGHTS.transforms


def extract_patch(vv_ds, vh_ds, x, y, size=PATCH_PX):
    row, col = vv_ds.index(x, y)
    r0, c0 = row - size // 2, col - size // 2
    if r0 < 0 or c0 < 0 or r0 + size > vv_ds.height or c0 + size > vv_ds.width:
        return None
    window = rasterio.windows.Window(c0, r0, size, size)
    vv = vv_ds.read(1, window=window)
    vh = vh_ds.read(1, window=window)
    if (vv == vv_ds.nodata).mean() > 0.3 or (vh == vh_ds.nodata).mean() > 0.3:
        return None
    vv = np.where(vv == vv_ds.nodata, np.nanmean(vv[vv != vv_ds.nodata]), vv)
    vh = np.where(vh == vh_ds.nodata, np.nanmean(vh[vh != vh_ds.nodata]), vh)
    return np.stack([vv, vh]).astype(np.float32)


def build_patch_dataset():
    """Réutilisée par os4_finetune.py pour garder exactement le même jeu de
    patches/étiquettes entre le probing gelé et le fine-tuning."""
    tbe = load_tbe_polygons()
    patches, labels = [], []
    for year in YEARS:
        vv_path, vh_path = PROC_DIR / f"vv_db_{year}.tif", PROC_DIR / f"vh_db_{year}.tif"
        if not vv_path.exists():
            continue
        with rasterio.open(vv_path) as vv_ds, rasterio.open(vh_path) as vh_ds:
            raster_crs = vv_ds.crs
            year_gdf = tbe[tbe["year"] == year].reset_index(drop=True)
            neg = negative_samples(year, year_gdf.geometry)
            combined = gpd.GeoDataFrame(
                pd.concat([year_gdf, neg], ignore_index=True), crs="EPSG:4326"
            ).to_crs(raster_crs)

            n_ok = 0
            for _, row in combined.iterrows():
                c = row.geometry.centroid
                patch = extract_patch(vv_ds, vh_ds, c.x, c.y)
                if patch is None:
                    continue
                patches.append(patch)
                labels.append(int(row.severity))
                n_ok += 1
        print(f"  {year}: {len(combined)} polygones -> {n_ok} patches valides")
    return patches, labels


def main():
    model, transforms = load_model()
    patches, labels = build_patch_dataset()

    print(f"\n{len(patches)} patches au total. Extraction des embeddings (ResNet50 SSL4EO-S12, gelé)...")
    X_img = torch.from_numpy(np.stack(patches))  # (N, 2, 64, 64)
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(X_img), 32):
            batch = transforms(X_img[i:i + 32])
            emb = model(batch)
            embeddings.append(emb.numpy())
    embeddings = np.concatenate(embeddings)
    y = np.array(labels)

    df_emb = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
    df_emb["severity"] = y
    out_csv = RESULT_DIR / "sar_defoliation_embeddings_ssl4eo.csv"
    df_emb.to_csv(out_csv, index=False)
    print(f"Embeddings -> {out_csv}  (shape={embeddings.shape})")

    clf = RandomForestClassifier(n_estimators=400, max_depth=6, class_weight="balanced", random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, embeddings, y, cv=cv)
    acc = accuracy_score(y, y_pred)
    print(f"\nExactitude Random Forest sur embeddings SSL4EO-S12 (5-fold) : {acc:.3f}  (n={len(y)})")
    print("Rappel (mêmes années/AOI, échantillons non nécessairement identiques) :")
    print("  amplitude brute (n=1329)                : 0.383")
    print("  amplitude anomalie-corrigée (n=1310)     : 0.407")
    print("  texture GLCM seule (n=431)               : 0.346")
    print("  amplitude+texture combinées (n=431)      : 0.397")


if __name__ == "__main__":
    main()
