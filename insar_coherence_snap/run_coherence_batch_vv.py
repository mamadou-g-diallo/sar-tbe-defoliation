"""
Calcul par lots de la cohérence interférométrique Sentinel-1 (SNAP GPT), sur
des paires consécutives d'une série temporelle SLC triée par date.

Origine : réécriture propre et documentée du script utilisé opérationnellement
à l'UM6P pour la cartographie de cohérence suite au séisme d'Al Haouz (Maroc,
sept. 2023). Corrige par rapport à la version d'origine :
  - chemins portables (plus de chemins Windows D:/... en dur) ;
  - extraction de la date par expression régulière plutôt que par position de
    caractère fixe (échoue explicitement si le nom de fichier ne correspond
    pas au format attendu, plutôt que de mal parser silencieusement) ;
  - subprocess.run(check=...) avec liste d'arguments plutôt que os.system()
    sur une chaîne (plus sûr, code de retour vérifié, erreurs journalisées) ;
  - CLI (argparse) au lieu de chemins codés en dur dans le script.

Prérequis : ESA SNAP installé avec le module Sentinel-1 Toolbox, et
l'exécutable `gpt` accessible (sur le PATH, ou fourni via --gpt-path).

Convention de nommage attendue pour les fichiers SLC (standard ESA) :
    S1{A,B,C}_IW_SLC__1S{DV,DH,SV,SH}_AAAAMMJJTHHMMSS_..._....zip (ou .SAFE)
La date est extraite par expression régulière (AAAAMMJJ après 'IW_SLC__1S??_').

Utilisation :
    python3 run_coherence_batch_vv.py \\
        --input-dir /chemin/vers/SLC/ \\
        --output-dir /chemin/vers/sortie/ \\
        --graph coherence_graph_vv.xml \\
        --burst-first 1 --burst-last 9

    # Sans exécuter gpt, juste pour vérifier les paires et commandes générées :
    python3 run_coherence_batch_vv.py --input-dir ... --output-dir ... --dry-run
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Exemple : S1A_IW_SLC__1SDV_20230908T182645_20230908T182712_050157_060A5C_1234
DATE_PATTERN = re.compile(r"IW_SLC__1S[DS][VH]_(\d{8})T\d{6}")


def extract_date(filename: str) -> str:
    """Retourne la date AAAAMMJJ trouvée dans le nom de fichier SLC, ou lève
    une erreur explicite si le nom ne correspond pas au format ESA attendu -
    plutôt que de mal parser silencieusement des positions de caractères."""
    m = DATE_PATTERN.search(filename)
    if not m:
        raise ValueError(
            f"Nom de fichier inattendu (format SLC ESA standard requis) : {filename!r}"
        )
    return m.group(1)


def find_gpt(explicit_path: str | None) -> str:
    if explicit_path:
        return explicit_path
    import shutil
    gpt = shutil.which("gpt")
    if gpt is None:
        sys.exit(
            "Exécutable 'gpt' introuvable sur le PATH. Installez ESA SNAP "
            "(https://step.esa.int/main/download/snap-download/) ou précisez "
            "--gpt-path /chemin/vers/gpt."
        )
    return gpt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, help="Dossier contenant les produits SLC (.zip/.SAFE)")
    parser.add_argument("--output-dir", required=True, help="Dossier de sortie pour les GeoTIFF de cohérence")
    parser.add_argument("--graph", default=str(Path(__file__).parent / "coherence_graph_vv.xml"),
                         help="Graphe SNAP GPT à utiliser (défaut : coherence_graph_vv.xml à côté de ce script)")
    parser.add_argument("--gpt-path", default=None, help="Chemin vers l'exécutable gpt (défaut : cherché sur le PATH)")
    parser.add_argument("--burst-first", type=int, default=1, help="Premier burst à extraire (dépend de l'AOI)")
    parser.add_argument("--burst-last", type=int, default=9, help="Dernier burst à extraire (dépend de l'AOI)")
    parser.add_argument("--xmx", default="16G", help="Mémoire max JVM (-J-Xmx)")
    parser.add_argument("--xms", default="8G", help="Mémoire initiale JVM (-J-Xms)")
    parser.add_argument("--dry-run", action="store_true", help="Affiche les paires/commandes sans exécuter gpt")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gpt = None if args.dry_run else find_gpt(args.gpt_path)

    files = []
    for f in input_dir.iterdir():
        if not f.is_file():
            continue
        try:
            extract_date(f.name)
        except ValueError:
            print(f"  [ignoré] {f.name} (ne correspond pas au format SLC attendu)")
            continue
        files.append(f)
    files.sort(key=lambda f: extract_date(f.name))
    if len(files) < 2:
        sys.exit(f"Au moins 2 produits SLC nécessaires dans {input_dir} (trouvé : {len(files)}).")

    print(f"{len(files)} produits SLC triés par date, {len(files) - 1} paires consécutives à traiter.")

    failures = []
    for f1, f2 in zip(files, files[1:]):
        d1, d2 = extract_date(f1.name), extract_date(f2.name)
        out_path = output_dir / f"Coherence_VV_{d1}_{d2}.tif"

        if out_path.exists():
            print(f"  [déjà fait] {out_path.name}")
            continue

        cmd = [
            gpt or "gpt", args.graph,
            f"-J-Xms{args.xms}", f"-J-Xmx{args.xmx}",
            f"-PInputFile1={f1}",
            f"-PInputFile2={f2}",
            f"-PBurstFirst={args.burst_first}",
            f"-PBurstLast={args.burst_last}",
            f"-POutputFile={out_path}",
        ]

        if args.dry_run:
            print(f"  [dry-run] {' '.join(cmd)}")
            continue

        print(f"  [en cours] {d1} -> {d2} ...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    ÉCHEC (code {result.returncode}) :\n{result.stderr[-2000:]}")
            failures.append((f1.name, f2.name))
        else:
            print(f"    OK -> {out_path.name}")

    if failures:
        print(f"\n{len(failures)} paire(s) en échec :")
        for a, b in failures:
            print(f"  {a} / {b}")
        sys.exit(1)

    print("\nTerminé, aucune erreur.")


if __name__ == "__main__":
    main()
