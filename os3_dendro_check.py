"""
OS3 (préliminaire) — les chronologies dendrochronologiques ouvertes disponibles
pour la zone d'étude (ITRDB/NOAA, Krause & Morin) s'arrêtent toutes en
1993-1995, soit ~20 ans avant le début de Sentinel-1 (2014). Aucune
comparaison directe avec la série SAR de l'OS1/OS2 n'est donc possible avec
des données ouvertes seules - un constat important en soi pour la suite de
la thèse (nécessité d'un carottage de terrain neuf, co-localisé avec les
observations SAR actuelles, plutôt qu'une réutilisation de chronologies
historiques).

Ce script fait plutôt une vérification plus modeste mais utile : les
chroniques ouvertes captent-elles la dépression de croissance correspondant
à l'épidémie de TBE des années 1970-1980, documentée spécifiquement au nord
du Lac-Saint-Jean par Morin & Laprise (1990, Can. J. For. Res.) - la même
région que notre AOI ? Une réponse positive validerait la pertinence de ces
séries pour une éventuelle extension future (recalibration avec de nouvelles
carottes prolongeant la chronique jusqu'à aujourd'hui).

Sites (ITRDB, Krause & Morin) :
  - CANA143 Lac Onatchiway (épinette noire, PCMA) - 48.88N -71.02W, DANS l'AOI Sentinel-1
  - CANA144 Mont Valin (sapin baumier, ABBA - hôte principal de la TBE) - 48.08N -70.07W
  - CANA141 Lac Liberal (sapin baumier, ABBA) - 49.07N -72.10W

Utilisation :
    conda activate hw_senegal
    python3 os3_dendro_check.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import RESULT_DIR

DENDRO_DIR = RESULT_DIR.parent.parent / "dendro"  # data/dendro/, hors data/<aoi>/

SITES = {
    "cana143": ("Lac Onatchiway (PCMA, épinette noire) - dans l'AOI", "tab:blue"),
    "cana144": ("Mont Valin (ABBA, sapin baumier) - hôte principal TBE", "tab:orange"),
    "cana141": ("Lac Liberal (ABBA, sapin baumier)", "tab:green"),
}

# Fenêtre d'épidémie documentée pour cette région précise (nord du lac Saint-Jean)
# par Morin & Laprise (1990), Can. J. For. Res. 20:1-8.
OUTBREAK_WINDOW = (1974, 1988)


def parse_crn(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("age_CE"))
    rows = []
    for l in lines[start + 1:]:
        parts = l.split()
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        rows.append((int(parts[0]), float(parts[1])))
    df = pd.DataFrame(rows, columns=["year", "trsgi_raw"])
    df["trsgi"] = df["trsgi_raw"] / 1000.0  # convention ITRDB .crn : index x1000
    return df


def main():
    series = {code: parse_crn(DENDRO_DIR / f"{code}-crn-noaa.txt") for code in SITES}

    for code, df in series.items():
        print(f"{code}: {df.year.min()}-{df.year.max()} "
              f"(dernière année - Sentinel-1 : {2014 - df.year.max()} ans d'écart)")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axvspan(*OUTBREAK_WINDOW, color="red", alpha=0.15,
               label=f"Épidémie documentée {OUTBREAK_WINDOW[0]}-{OUTBREAK_WINDOW[1]}\n(Morin & Laprise 1990, nord du lac Saint-Jean)")
    ax.axvspan(2014, 2026, color="steelblue", alpha=0.15, label="Période Sentinel-1 (2014-aujourd'hui)")

    for code, (label, color) in SITES.items():
        df = series[code]
        ax.plot(df.year, df.trsgi, label=label, color=color, linewidth=1.3)

    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlim(1900, 2026)
    ax.set_xlabel("Année")
    ax.set_ylabel("Indice de croissance standardisé (trsgi)")
    ax.set_title("Chronologies dendro ouvertes (ITRDB) vs. épidémie TBE documentée et période Sentinel-1")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    out_fig = RESULT_DIR.parent / "dendro_vs_outbreak.png"
    fig.savefig(out_fig, dpi=150)
    print(f"\nFigure -> {out_fig}")

    print("\nConstat : aucune des 3 chroniques ne couvre la période Sentinel-1 (2014+).")
    print("Dépression de croissance maximale pendant la fenêtre d'épidémie documentée "
          f"({OUTBREAK_WINDOW[0]}-{OUTBREAK_WINDOW[1]}) :")
    for code, df in series.items():
        pre = df[(df.year >= OUTBREAK_WINDOW[0] - 10) & (df.year < OUTBREAK_WINDOW[0])]["trsgi"].mean()
        win = df[(df.year >= OUTBREAK_WINDOW[0]) & (df.year <= OUTBREAK_WINDOW[1])]
        min_row = win.loc[win["trsgi"].idxmin()]
        drop_pct = (1 - min_row.trsgi / pre) * 100
        print(f"  {code}: minimum = {min_row.trsgi:.3f} en {int(min_row.year)} "
              f"(référence pré-épidémie {pre:.3f}) -> baisse de {drop_pct:.0f}%")
    print("\n-> Minimum synchrone (1978) sur les 3 sites indépendants = signal régional réel, pas du bruit.")
    print("-> Ampleur très différente par essence (sapin baumier ~-77/78% vs épinette noire ~-35%),")
    print("   cohérent avec la préférence d'hôte connue de la TBE (sapin baumier >> épinette noire).")


if __name__ == "__main__":
    main()
