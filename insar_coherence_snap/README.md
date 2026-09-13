# Cohérence interférométrique Sentinel-1 (SNAP GPT)

Chaîne de calcul de cohérence InSAR par lots, sur une série temporelle de
produits SLC Sentinel-1. **Séparée du reste de ce dépôt** : elle utilise
[ESA SNAP](https://step.esa.int/main/download/snap-download/) (Java,
installation locale requise), pas la pile Python/rasterio du pipeline
principal.

## Origine et contexte

Nettoyé et documenté à partir du graphe et du script utilisés
opérationnellement à l'UM6P (dans le cadre du poste Remote Sensing & GIS
Engineer, sept. 2023 – mars 2025) pour cartographier la cohérence et le
déplacement du sol suite au **séisme d'Al Haouz** (Maroc, 8 septembre 2023),
à la demande du responsable du laboratoire.

Repris ici pour deux raisons :
1. le rendre réutilisable sans ambiguïté (noms de nœuds explicites, paramètres
   externalisés au lieu d'être codés en dur, choix méthodologiques commentés) ;
2. fournir l'outil qui manquait à l'exploration de la cohérence InSAR pour la
   défoliation TBE (cf. section OS2 du README principal) — dès que des paires
   SLC sur l'AOI Saguenay–Lac-Saint-Jean sont disponibles (l'acquisition ASF
   avait été mise en pause, cf. historique du projet), cet outil calcule la
   cohérence sans dépendre d'une librairie tierce non vérifiée.

## Chaîne de traitement

```
Read (x2, master+esclave)
  -> Apply-Orbit-File (orbites précises, téléchargement automatique)
  -> TOPSAR-Split (IW1, IW2, IW3 — x2 images chacune)
  -> Back-Geocoding (coregistrement, DEM SRTM 3 arcsec)
  -> Coherence (fenêtre 2 az. x 10 rg., par sous-fauchée)
  -> TOPSAR-Deburst (par sous-fauchée)
  -> TOPSAR-Merge (fusion IW1+IW2+IW3)
  -> Terrain-Correction (géocodage WGS84, ~14 m)
  -> Write (GeoTIFF)
```

## Limite méthodologique assumée

Le graphe calcule une cohérence **sans retrait de la rampe de phase**
(`subtractFlatEarthPhase=false`, `subtractTopographicPhase=false`). C'est
suffisant pour une carte de cohérence (détection de changement de surface),
et c'est le produit que ce graphe a été conçu pour fournir en lot sur une
série temporelle complète. **Ce n'est pas suffisant pour extraire un
déplacement en ligne de visée (LOS)** à partir de la phase : cela demande en
plus le déroulement de phase (SNAPHU) et la conversion géométrique LOS, non
inclus ici et généralement fait au cas par cas sur la paire d'intérêt plutôt
qu'en lot.

Autre limite : sans retrait de la rampe de phase, la cohérence peut être
légèrement sous-estimée dans les zones à fort relief ou grande ligne de base
(décorrélation géométrique confondue avec une vraie décorrélation temporelle)
— pertinent pour un relief marqué comme l'Atlas, moins pour une zone plane.

## Prérequis

- [ESA SNAP](https://step.esa.int/main/download/snap-download/) avec le
  module Sentinel-1 Toolbox installé, exécutable `gpt` accessible (sur le
  `PATH`, ou fourni via `--gpt-path`)
- Accès Internet (téléchargement automatique des orbites précises et du DEM
  SRTM au premier lancement)
- Python 3 (script de lancement uniquement — aucune dépendance externe)

## Utilisation

```bash
python3 run_coherence_batch.py \
    --input-dir /chemin/vers/SLC/ \
    --output-dir /chemin/vers/sortie/ \
    --burst-first 1 --burst-last 9
```

Génère une cohérence VV pour chaque **paire consécutive** de la série
temporelle triée par date (stratégie standard pour un suivi de cohérence
dans le temps). Reprise automatique : une paire déjà calculée (fichier de
sortie déjà présent) est sautée.

Pour vérifier les paires et commandes générées sans exécuter SNAP :
```bash
python3 run_coherence_batch.py --input-dir ... --output-dir ... --dry-run
```

### Trouver la plage de bursts (`--burst-first`/`--burst-last`)

Dépend de l'AOI et doit être redéterminée pour chaque nouvelle zone d'étude :
ouvrir un des produits SLC dans SNAP Desktop (Graph Builder ou simple
ouverture), afficher la sous-fauchée concernée, et lire les index de burst
qui couvrent l'AOI dans le panneau d'information du produit — ou utiliser
`TOPSAR-Split` en mode interactif une fois pour identifier visuellement la
plage avant de la réutiliser en lot.

## Fichiers

- [`coherence_graph.xml`](coherence_graph.xml) — graphe SNAP GPT (paramètres :
  `InputFile1`, `InputFile2`, `OutputFile`, `BurstFirst`, `BurstLast`)
- [`run_coherence_batch.py`](run_coherence_batch.py) — script de lancement en
  lot sur une série temporelle SLC
