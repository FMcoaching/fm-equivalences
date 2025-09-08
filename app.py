from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import difflib

from fm_equivalences import load_food_db, equivalent_portion  # on garde tes routes existantes

CSV = "nutrient_values_clean.csv"
db = load_food_db(CSV)  # contient déjà les colonnes par 100 g: kcal, protein, carb, fat, fiber
FOODS = sorted(db["Aliment"].dropna().unique().tolist())

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


# --------- utilitaires ----------
def best_match(name: str) -> Optional[str]:
    if not name:
        return None
    contains = [x for x in FOODS if name.lower() in x.lower()]
    if contains:
        return contains[0]
    close = difflib.get_close_matches(name, FOODS, n=1, cutoff=0.5)
    return close[0] if close else None

def row_for(name: str):
    """Retourne la ligne db (Series) pour un aliment via best_match, ou None."""
    m = best_match(name)
    if not m:
        return None
    rows = db[db["Aliment"] == m]
    return rows.iloc[0] if not rows.empty else None

def macros_for_grams(row, grams: float) -> Dict[str, float]:
    f = grams / 100.0
    return {
        "kcal": round(float(row["kcal_per_100g"]) * f, 4),
        "prot_g": round(float(row["protein_g_per_100g"]) * f, 4),
        "carb_g": round(float(row["carb_g_per_100g"]) * f, 4),
        "fat_g":  round(float(row["fat_g_per_100g"]) * f, 4),
        "fiber_g":round(float(row["fiber_g_per_100g"]) * f, 4),
    }

def totals_for_plan(items: List[Dict[str, float]]) -> Dict[str, float]:
    t = {"kcal": 0.0, "prot_g": 0.0, "carb_g": 0.0, "fat_g": 0.0, "fiber_g": 0.0}
    for it in items:
        r = row_for(it["aliment"])
        if r is None:
            # On ignore les inconnus (ou on pourrait lever une erreur)
            continue
        m = macros_for_grams(r, float(it["grams"]))
        for k in t.keys():
            t[k] += m[k]
    # arrondis d'affichage
    for k in t.keys():
        t[k] = round(t[k], 2)
    return t

def per_gram(row) -> Dict[str, float]:
    return {
        "p": float(row["protein_g_per_100g"]) / 100.0,
        "k": float(row["kcal_per_100g"]) / 100.0,
    }

def solve_2x2(p1, p2, k1, k2, dp, dk):
    """Résout:
       p1*x + p2*y = dp
       k1*x + k2*y = dk
       Renvoie (x,y) ou None si système quasi singulier.
    """
    d = p1*k2 - p2*k1
    if abs(d) < 1e-9:
        return None
    x = (dp*k2 - dk*p2) / d
    y = (p1*dk - k1*dp) / d
    return (x, y)

def pick_adjustables(target_items: List[Dict[str, float]]):
    """Choisit automatiquement 2 aliments à ajuster:
       - un le plus 'protéiné' (densité protéique)
       - un le plus 'énergétique' avec ratio protéines/kcal très différent
    """
    # liste de (name, grams, row, p_per_g, k_per_g, ratio)
    cand = []
    for it in target_items:
        r = row_for(it["aliment"])
        if r is None:
            continue
        pg = float(r["protein_g_per_100g"]) / 100.0
        kg = float(r["kcal_per_100g"]) / 100.0
        if pg == 0 and kg == 0:
            continue
        ratio = pg / kg if kg != 0 else float("inf")
        cand.append((best_match(it["aliment"]), float(it["grams"]), r, pg, kg, ratio))
    if len(cand) < 2:
        return None

    # 1) meilleur "prot" = max pg
    prot = max(cand, key=lambda x: x[3])

    # 2) meilleur "énergie" = ratio le plus différent du prot
    cand2 = [c for c in cand if c[0] != prot[0]]
    if not cand2:
        return None
    energy = max(cand2, key=lambda x: abs(x[5] - prot[5]))

    return (prot, energy)  # tuples complets


# --------- routes existantes ----------
@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")

@app.get("/search")
def search(q: str):
    hits = [x for x in FOODS if q.lower() in x.lower()][:8]
    if not hits:
        hits = difflib.get_close_matches(q, FOODS, n=8, cutoff=0.5)
    return {"results": hits}

class EqReq(BaseModel):
    source: str
    grams: float
    target: str
    mode: str = "carbs"  # "carbs" ou "kcal"

@app.post("/equivalence")
def equivalence(req: EqReq):
    try:
        res = equivalent_portion(db, req.source, req.grams, req.target, mode=req.mode)
        res["target_grams"] = round(res["target_grams"] / 5) * 5
        return res
    except Exception as e:
        return {"error": str(e)}


# --------- NOUVEAU: comparer 2 plans, iso-protéines ET iso-calories ----------
class PlanItem(BaseModel):
    aliment: str
    grams: float

class CompareReq(BaseModel):
    current: List[PlanItem]
    target: List[PlanItem]
    adjustable: Optional[List[str]] = None  # optionnel: 2 noms à ajuster
    round_to: int = 5  # pas d'arrondi des grammes suggérés

@app.post("/compare-plans")
def compare_plans(req: CompareReq):
    # Totaux actuels vs cibles initiales
    cur_list = [it.dict() for it in req.current]
    tgt_list = [it.dict() for it in req.target]
    cur_tot = totals_for_plan(cur_list)
    tgt_tot = totals_for_plan(tgt_list)

    # Deltas à combler (prot et kcal EXACTS)
    dp = round(cur_tot["prot_g"] - tgt_tot["prot_g"], 6)
    dk = round(cur_tot["kcal"]  - tgt_tot["kcal"],  6)

    # Choix des 2 aliments à ajuster
    if req.adjustable and len(req.adjustable) >= 2:
        # Utiliser exactement les deux fournis par le coach/client
        a1_name = best_match(req.adjustable[0])
        a2_name = best_match(req.adjustable[1])
        if not a1_name or not a2_name:
            return {"error": "Impossible de trouver un des aliments 'adjustable' dans la base."}
        # trouver grams actuels dans target (0 si absent)
        def grams_in_target(nm):
            for it in tgt_list:
                m = best_match(it["aliment"])
                if m == nm:
                    return float(it["grams"])
            return 0.0
        g1 = grams_in_target(a1_name)
        g2 = grams_in_target(a2_name)
        r1 = row_for(a1_name); r2 = row_for(a2_name)
        p1k1 = per_gram(r1); p2k2 = per_gram(r2)
        sol = solve_2x2(p1k1["p"], p2k2["p"], p1k1["k"], p2k2["k"], dp, dk)
        if sol is None:
            return {"error": "Conditions incompatibles avec les 2 aliments choisis (ratios très proches). Choisissez un aliment riche en protéines et un autre riche en énergie."}
        dx, dy = sol
        new_g1 = g1 + dx
        new_g2 = g2 + dy
        # arrondi pratique
        rt = max(1, int(req.round_to))
        new_g1 = round(new_g1 / rt) * rt
        new_g2 = round(new_g2 / rt) * rt

        # reconstruire un plan ajusté (les autres aliments identiques)
        adjusted = []
        used = set()
        for it in tgt_list:
            m = best_match(it["aliment"])
            if m == a1_name and m not in used:
                adjusted.append({"aliment": a1_name, "grams": max(0.0, new_g1)})
                used.add(m)
            elif m == a2_name and m not in used:
                adjusted.append({"aliment": a2_name, "grams": max(0.0, new_g2)})
                used.add(m)
            else:
                adjusted.append({"aliment": it["aliment"], "grams": it["grams"]})
        # si un des ajustables était absent au départ, on l'ajoute
        if a1_name not in [best_match(x["aliment"]) for x in tgt_list]:
            adjusted.append({"aliment": a1_name, "grams": max(0.0, new_g1)})
        if a2_name not in [best_match(x["aliment"]) for x in tgt_list]:
            adjusted.append({"aliment": a2_name, "grams": max(0.0, new_g2)})

    else:
        # Auto-pick: 1 aliment très protéiné + 1 très énergétique
        picked = pick_adjustables(tgt_list)
        if picked is None:
            return {"error": "Plan cible insuffisant pour ajustement (il faut au moins 2 aliments reconnus)."}
        (a1_name, g1, r1, p1, k1, rratio1), (a2_name, g2, r2, p2, k2, rratio2) = picked
        sol = solve_2x2(p1, p2, k1, k2, dp, dk)
        if sol is None:
            return {"error": "Aliments cibles choisis automatiquement non compatibles (ratios trop proches). Spécifiez 2 'adjustable' différents (ex: un très protéiné + un énergétique)."}
        dx, dy = sol
        rt = max(1, int(req.round_to))
        new_g1 = round((g1 + dx) / rt) * rt
        new_g2 = round((g2 + dy) / rt) * rt

        # reconstruire le plan ajusté
        adjusted = []
        used = set()
        for it in tgt_list:
            m = best_match(it["aliment"])
            if m == a1_name and m not in used:
                adjusted.append({"aliment": a1_name, "grams": max(0.0, new_g1)})
                used.add(m)
            elif m == a2_name and m not in used:
                adjusted.append({"aliment": a2_name, "grams": max(0.0, new_g2)})
                used.add(m)
            else:
                adjusted.append({"aliment": it["aliment"], "grams": it["grams"]})

    # Totaux finaux (après ajustement)
    final_tot = totals_for_plan(adjusted)
    diffs = {
        "kcal": round(cur_tot["kcal"] - final_tot["kcal"], 2),
        "prot_g": round(cur_tot["prot_g"] - final_tot["prot_g"], 2),
        "carb_g": round(cur_tot["carb_g"] - final_tot["carb_g"], 2),
        "fat_g":  round(cur_tot["fat_g"]  - final_tot["fat_g"],  2),
    }

    return {
        "current_totals": cur_tot,
        "initial_target_totals": tgt_tot,
        "needed_diff": {"kcal": dk, "prot_g": dp},
        "adjusted_plan": adjusted,
        "final_totals": final_tot,
        "residual_diff": diffs,  # devrait être (0,0,*,*) à l'arrondi près pour kcal/protéines
    }
