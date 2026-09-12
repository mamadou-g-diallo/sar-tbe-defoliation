"""
OS4 (suite) — fine-tuning partiel du modèle fondation SSL4EO-S12 (au lieu du
simple "linear probing" de os4_foundation_model_probe.py) : le backbone
ResNet50 pré-entraîné (MoCo, Sentinel-1 VV/VH) garde ses couches précoces
gelées (features SAR bas niveau, probablement déjà génériques), mais
layer4 + une nouvelle tête de classification sont ré-entraînés sur nos
échantillons de sévérité TBE.

Contexte : le probing gelé (embeddings + Random Forest) atteignait déjà
64,0% d'exactitude contre 38,3% pour l'amplitude seule (cf. README). Ce
script teste si dégeler la dernière partie du réseau, plutôt que de traiter
ses features comme fixes, améliore encore la séparation - au prix d'un
entraînement plus coûteux (rétropropagation) et d'un risque de surapprentissage
plus élevé sur un jeu d'à peine ~1400 échantillons.

Utilisation :
    conda activate hw_senegal
    TBE_AOI=<nom_aoi> python3 os4_finetune.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchgeo.models import ResNet50_Weights, resnet50

from config import RESULT_DIR, SEVERITY_LABELS
from os4_foundation_model_probe import build_patch_dataset, WEIGHTS

N_CLASSES = 4
EPOCHS = 20
BATCH_SIZE = 16
LR = 1e-4
PATIENCE = 5
SEED = 42


class PatchDataset(Dataset):
    def __init__(self, patches, labels, transforms, augment=False):
        self.patches = patches
        self.labels = labels
        self.transforms = transforms
        self.augment = augment

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.patches[idx].copy())
        if self.augment:
            if np.random.rand() < 0.5:
                x = torch.flip(x, dims=[1])
            if np.random.rand() < 0.5:
                x = torch.flip(x, dims=[2])
            if np.random.rand() < 0.5:
                x = torch.rot90(x, k=1, dims=[1, 2])
        x = self.transforms(x.unsqueeze(0)).squeeze(0)
        return x, self.labels[idx]


def build_model():
    model = resnet50(weights=WEIGHTS)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer4.parameters():
        p.requires_grad = True
    model.fc = nn.Linear(2048, N_CLASSES)  # nouvelle tête, entraînable par construction
    return model


def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, all_preds, all_labels = 0.0, [], []
    for x, y in loader:
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(y)
        all_preds.append(out.argmax(1).detach().numpy())
        all_labels.append(y.numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc = (preds == labels).mean()
    return total_loss / len(labels), acc


def main():
    torch.manual_seed(SEED)
    print("Extraction des patches (identique à os4_foundation_model_probe.py)...")
    patches, labels = build_patch_dataset()
    patches = np.stack(patches)
    labels = np.array(labels)
    print(f"{len(labels)} patches, distribution : {dict(zip(*np.unique(labels, return_counts=True)))}")

    idx = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(idx, test_size=0.3, stratify=labels, random_state=SEED)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=SEED)
    print(f"Train={len(train_idx)}  Val={len(val_idx)}  Test={len(test_idx)}")

    model = build_model()
    transforms = WEIGHTS.transforms

    train_ds = PatchDataset(patches[train_idx], labels[train_idx], transforms, augment=True)
    val_ds = PatchDataset(patches[val_idx], labels[val_idx], transforms, augment=False)
    test_ds = PatchDataset(patches[test_idx], labels[test_idx], transforms, augment=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

    class_counts = np.bincount(labels[train_idx], minlength=N_CLASSES)
    weights = torch.tensor(1.0 / np.maximum(class_counts, 1), dtype=torch.float32)
    weights = weights / weights.sum() * N_CLASSES
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss, best_state, patience_left = np.inf, None, PATIENCE

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_dl, criterion)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        print(f"  epoch {epoch:2d}/{EPOCHS}  train_loss={tr_loss:.3f} acc={tr_acc:.3f}  "
              f"val_loss={val_loss:.3f} acc={val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss, best_state, patience_left = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, PATIENCE
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"  arrêt anticipé (pas d'amélioration en validation depuis {PATIENCE} époques)")
                break

    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_dl:
            out = model(x)
            all_preds.append(out.argmax(1).numpy())
            all_labels.append(y.numpy())
    preds = np.concatenate(all_preds)
    truth = np.concatenate(all_labels)

    print(f"\nExactitude test (fine-tuning layer4+fc) : {(preds == truth).mean():.3f}  (n={len(truth)})")
    print(classification_report(truth, preds, target_names=[SEVERITY_LABELS[c] for c in sorted(set(truth))]))
    print("Matrice de confusion :")
    print(confusion_matrix(truth, preds))
    print("\nRappel : linear probing (embeddings gelés + RF, n=1413) = 0.640")
    print("Rappel : amplitude seule (RF, n=1329) = 0.383")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Perte (cross-entropy)")
    axes[0].set_xlabel("Époque")
    axes[0].legend()
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Exactitude")
    axes[1].set_xlabel("Époque")
    axes[1].legend()
    fig.tight_layout()
    out_fig = RESULT_DIR / "os4_finetune_curves.png"
    fig.savefig(out_fig, dpi=150)
    print(f"\nFigure -> {out_fig}")


if __name__ == "__main__":
    main()
