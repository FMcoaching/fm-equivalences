
import pandas as pd
import numpy as np
import difflib
import re

def load_food_db(csv_path):
    df = pd.read_csv(csv_path)
    def norm_name(x):
        if isinstance(x, str):
            x = re.sub(r"\s+", " ", x.strip())
        return x
    for col in ["Aliment", "Mesure"]:
        if col in df.columns:
            df[col] = df[col].apply(norm_name)
    num_cols = [c for c in df.columns if c not in ["Aliment","Mesure"]]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["Aliment"].notna() & df["Poids (g)"].notna() & (df["Poids (g)"] > 0)]
    per100 = df.copy()
    factor = 100.0 / per100["Poids (g)"]
    per100["kcal_per_100g"] = per100["Énergie (kcal)"] * factor
    per100["protein_g_per_100g"] = per100["Protéines (g)"] * factor
    per100["carb_g_per_100g"] = per100["Glucides (g)"] * factor
    per100["fat_g_per_100g"] = per100["Lipides (g)"] * factor
    per100["fiber_g_per_100g"] = per100["Fibres (g)"] * factor
    for c in ["kcal_per_100g","protein_g_per_100g","carb_g_per_100g","fat_g_per_100g","fiber_g_per_100g"]:
        per100[c] = per100[c].fillna(0.0)
    per100.reset_index(drop=True, inplace=True)
    return per100

def _choices(series, query, n=8):
    names = series.dropna().unique().tolist()
    contains = [x for x in names if query.lower() in x.lower()]
    close = difflib.get_close_matches(query, names, n=n, cutoff=0.5)
    seen = set(); out = []
    for x in contains + close:
        if x not in seen:
            out.append(x); seen.add(x)
    return out[:n]

def equivalent_portion(db, source_name, source_grams, target_name, mode="carbs", round_to=5):
    src_matches = _choices(db["Aliment"], source_name, n=1)
    tgt_matches = _choices(db["Aliment"], target_name, n=1)
    if not src_matches or not tgt_matches:
        raise ValueError("Aliment source ou cible introuvable dans la base.")
    src = db[db["Aliment"] == src_matches[0]].iloc[0]
    tgt = db[db["Aliment"] == tgt_matches[0]].iloc[0]
    src_carb_per_g = src["carb_g_per_100g"] / 100.0
    tgt_carb_per_g = tgt["carb_g_per_100g"] / 100.0
    src_kcal_per_g = src["kcal_per_100g"] / 100.0
    tgt_kcal_per_g = tgt["kcal_per_100g"] / 100.0
    if mode == "carbs":
        src_basis = src_carb_per_g * source_grams
        if tgt_carb_per_g == 0:
            raise ValueError("Cible 0 g glucides/100 g — iso-glucides impossible.")
        target_grams = src_basis / tgt_carb_per_g
    elif mode == "kcal":
        src_basis = src_kcal_per_g * source_grams
        if tgt_kcal_per_g == 0:
            raise ValueError("Cible 0 kcal/100 g — iso-kcal impossible.")
        target_grams = src_basis / tgt_kcal_per_g
    else:
        raise ValueError("mode doit être 'carbs' ou 'kcal'.")
    if round_to:
        target_grams = round(float(target_grams) / round_to) * round_to
    def macros_for(g, row):
        factor = g / 100.0
        return {
            "kcal": round(row["kcal_per_100g"] * factor, 1),
            "prot_g": round(row["protein_g_per_100g"] * factor, 1),
            "carb_g": round(row["carb_g_per_100g"] * factor, 1),
            "fat_g": round(row["fat_g_per_100g"] * factor, 1),
            "fiber_g": round(row["fiber_g_per_100g"] * factor, 1),
        }
    src_mac = macros_for(source_grams, src)
    tgt_mac = macros_for(target_grams, tgt)
    return {
        "mode": mode,
        "source_name": src["Aliment"],
        "source_grams": source_grams,
        "target_name": tgt["Aliment"],
        "target_grams": target_grams,
        "source_macros": src_mac,
        "target_macros": tgt_mac,
        "source_mesure": src["Mesure"],
        "target_mesure": tgt["Mesure"]
    }
