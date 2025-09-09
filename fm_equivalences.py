# fm_equivalences.py — version unifiée (Solution A par défaut + rétro-compat)
import pandas as pd
import numpy as np
import difflib
import re

# Colonnes attendues pour un CSV déjà normalisé "par 100 g"
REQUIRED_PER100 = [
    "Aliment",
    "kcal_per_100g",
    "protein_g_per_100g",
    "carb_g_per_100g",
    "fat_g_per_100g",
    "fiber_g_per_100g",
]

# Colonnes d'un CSV "ancien format" (style CNF/FCE) à convertir
LEGACY_COLS = {
    "name": "Aliment",
    "weight": "Poids (g)",
    "kcal": "Énergie (kcal)",
    "protein": "Protéines (g)",
    "carb": "Glucides (g)",
    "fat": "Lipides (g)",
    "fiber": "Fibres (g)",
    # "Mesure" peut exister ou pas; on l'utilise si présent
}

def _norm_name(x):
    if isinstance(x, str):
        x = re.sub(r"\s+", " ", x.strip())
    return x

def _to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _has_per100(df) -> bool:
    return all(c in df.columns for c in REQUIRED_PER100)

def _has_legacy(df) -> bool:
    need = [LEGACY_COLS["name"], LEGACY_COLS["weight"],
            LEGACY_COLS["kcal"], LEGACY_COLS["protein"],
            LEGACY_COLS["carb"], LEGACY_COLS["fat"], LEGACY_COLS["fiber"]]
    return all(c in df.columns for c in need)

def load_food_db(csv_path: str) -> pd.DataFrame:
    """
    Charge un CSV de données alimentaires.
    - Chemin "Solution A" (recommandé) : CSV déjà standardisé 'par 100 g' :
        Colonnes: Aliment,kcal_per_100g,protein_g_per_100g,carb_g_per_100g,fat_g_per_100g,fiber_g_per_100g
    - Rétro-compatibilité : si le CSV est 'ancien format' (avec Poids (g), Énergie (kcal), ...),
      on convertit automatiquement en per-100g.
    Retourne un DataFrame propre, typé, et sans NaN (0.0 pour numériques).
    """
    df = pd.read_csv(csv_path)

    # Normalisation simple des champs texte si présents
    for col in ["Aliment", "Mesure"]:
        if col in df.columns:
            df[col] = df[col].apply(_norm_name)

    # Chemin 1 : déjà per-100g (Solution A)
    if _has_per100(df):
        # Nettoyage/typage
        df["Aliment"] = df["Aliment"].astype(str).str.strip()
        num_cols = [c for c in REQUIRED_PER100 if c != "Aliment"]
        df = _to_numeric(df, num_cols)
        for c in num_cols:
            df[c] = df[c].fillna(0.0)
        # Déduplication
        df = df.drop_duplicates(subset=["Aliment"], keep="first").reset_index(drop=True)
        return df

    # Chemin 2 : ancien format → conversion per-100g
    if _has_legacy(df):
        # Typage numérique des colonnes utiles
        num_cols = [
            LEGACY_COLS["weight"],
            LEGACY_COLS["kcal"],
            LEGACY_COLS["protein"],
            LEGACY_COLS["carb"],
            LEGACY_COLS["fat"],
            LEGACY_COLS["fiber"],
        ]
        df = _to_numeric(df, num_cols)

        # Lignes valides (nom + poids > 0)
        df = df[
            df[LEGACY_COLS["name"]].notna()
            & df[LEGACY_COLS["weight"]].notna()
            & (df[LEGACY_COLS["weight"]] > 0)
        ].copy()

        # Conversion per-100g
        factor = 100.0 / df[LEGACY_COLS["weight"]]
        per100 = pd.DataFrame({
            "Aliment": df[LEGACY_COLS["name"]].astype(str).str.strip(),
            "kcal_per_100g": df[LEGACY_COLS["kcal"]] * factor,
            "protein_g_per_100g": df[LEGACY_COLS["protein"]] * factor,
            "carb_g_per_100g": df[LEGACY_COLS["carb"]] * factor,
            "fat_g_per_100g": df[LEGACY_COLS["fat"]] * factor,
            "fiber_g_per_100g": df[LEGACY_COLS["fiber"]] * factor,
        })

        # Nettoyage final
        for c in REQUIRED_PER100:
            if c != "Aliment":
                per100[c] = pd.to_numeric(per100[c], errors="coerce").fillna(0.0)
        per100["Aliment"] = per100["Aliment"].apply(_norm_name)

        # Déduplication
        per100 = per100.drop_duplicates(subset=["Aliment"], keep="first").reset_index(drop=True)
        return per100

    # Si ni per-100g ni legacy : erreur claire
    raise ValueError(
        "Format CSV non reconnu. Fournis soit un CSV déjà 'par 100 g' "
        f"avec colonnes {REQUIRED_PER100}, soit un ancien CSV CNF/FCE avec "
        f"les colonnes attendues (ex.: {list(LEGACY_COLS.values())})."
    )

def _choices(series: pd.Series, query: str, n: int = 8):
    names = series.dropna().astype(str).unique().tolist()
    q = (query or "").strip()
    if not q:
        return names[:n]
    contains = [x for x in names if q.lower() in x.lower()]
    close = difflib.get_close_matches(q, names, n=n, cutoff=0.5)
    seen = set(); out = []
    for x in contains + close:
        if x not in seen:
            out.append(x); seen.add(x)
    return out[:n]

def equivalent_portion(db: pd.DataFrame,
                       source_name: str,
                       source_grams: float,
                       target_name: str,
                       mode: str = "carbs",
                       round_to: int = 5):
    """
    Calcule la portion cible équivalente à partir d'une source.
    - 'mode' = 'carbs' (iso-glucides) ou 'kcal' (iso-calories)
    - Le DataFrame 'db' doit déjà contenir les colonnes per-100g.
    - 'Mesure' est optionnelle; si absente, on renvoie ''.
    """
    if not isinstance(source_grams, (int, float)) or source_grams <= 0:
        raise ValueError("La quantité source doit être un nombre positif (en g).")

    # Sélectionne la meilleure correspondance (source/target)
    src_matches = _choices(db["Aliment"], source_name, n=1)
    tgt_matches = _choices(db["Aliment"], target_name, n=1)
    if not src_matches or not tgt_matches:
        raise ValueError("Aliment source ou cible introuvable dans la base.")

    src = db[db["Aliment"] == src_matches[0]].iloc[0]
    tgt = db[db["Aliment"] == tgt_matches[0]].iloc[0]

    # Ratios par gramme
    src_carb_per_g = float(src["carb_g_per_100g"]) / 100.0
    tgt_carb_per_g = float(tgt["carb_g_per_100g"]) / 100.0
    src_kcal_per_g = float(src["kcal_per_100g"]) / 100.0
    tgt_kcal_per_g = float(tgt["kcal_per_100g"]) / 100.0

    if mode == "carbs":
        basis = src_carb_per_g * float(source_grams)
        if tgt_carb_per_g == 0:
            raise ValueError("Cible 0 g glucides/100 g — iso-glucides impossible.")
        target_grams = basis / tgt_carb_per_g
    elif mode == "kcal":
        basis = src_kcal_per_g * float(source_grams)
        if tgt_kcal_per_g == 0:
            raise ValueError("Cible 0 kcal/100 g — iso-kcal impossible.")
        target_grams = basis / tgt_kcal_per_g
    else:
        raise ValueError("mode doit être 'carbs' ou 'kcal'.")

    # Arrondi pratique
    if round_to:
        target_grams = round(float(target_grams) / round_to) * round_to

    def macros_for(g: float, row: pd.Series):
        factor = float(g) / 100.0
        def safe(col):
            v = float(row.get(col, 0.0) or 0.0)
            return v
        return {
            "kcal":   round(safe("kcal_per_100g")   * factor, 1),
            "prot_g": round(safe("protein_g_per_100g") * factor, 1),
            "carb_g": round(safe("carb_g_per_100g") * factor, 1),
            "fat_g":  round(safe("fat_g_per_100g")  * factor, 1),
            "fiber_g":round(safe("fiber_g_per_100g")* factor, 1),
        }

    src_mac = macros_for(source_grams, src)
    tgt_mac = macros_for(target_grams, tgt)

    return {
        "mode": mode,
        "source_name": str(src["Aliment"]),
        "source_grams": float(source_grams),
        "target_name": str(tgt["Aliment"]),
        "target_grams": float(target_grams),
        "source_macros": src_mac,
        "target_macros": tgt_mac,
        "source_mesure": str(src.get("Mesure", "") or ""),
        "target_mesure": str(tgt.get("Mesure", "") or "")
    }
