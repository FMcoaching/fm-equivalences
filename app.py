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


# --------- NOUVEAU v2: comparer 2 plans, iso-protéines ET iso-calories avec bornes ----------
import numpy as np

class PlanItem(BaseModel):
    aliment: str
    grams: float

class AdjustableItem(BaseModel):
    name: str
    min_g: float = 0.0
    max_g: float | None = None  # None = pas de maximum

class CompareReq(BaseModel):
    current: List[PlanItem]
    target: List[PlanItem]
    adjustables: Optional[List[AdjustableItem]] = None  # 2 ou 3 recommandés
    round_to: int = 5  # arrondi final des grammes
    priority: str = "kcal"  # "kcal" ou "protein" pour le micro-ajustement final

def _p_k_per_g(row):
    return (
        float(row["protein_g_per_100g"]) / 100.0,
        float(row["kcal_per_100g"]) / 100.0,
    )

def _clip(v, lo, hi):
    if hi is None: return max(lo, v)
    return min(max(lo, v), hi)

@app.post("/compare-plans")
def compare_plans(req: CompareReq):
    lazy_load()
    if LOAD_ERR:
        return JSONResponse({"error": LOAD_ERR}, status_code=500)

    # 1) Totaux actuel vs cible initiale
    cur_list = [it.dict() for it in req.current]
    tgt_list = [it.dict() for it in req.target]
    cur_tot = totals_for_plan(cur_list)
    tgt_tot = totals_for_plan(tgt_list)

    # 2) Besoin à combler (exact sur prot + kcal)
    dp = round(cur_tot["prot_g"] - tgt_tot["prot_g"], 6)
    dk = round(cur_tot["kcal"]  - tgt_tot["kcal"],  6)

    # 3) Préparer les ajustables
    #    - si l'utilisateur n'en donne pas, auto-pick (2 éléments très différents)
    if not req.adjustables or len(req.adjustables) < 2:
        picked = pick_adjustables(tgt_list)
        if picked is None:
            return {"error": "Plan cible insuffisant pour ajustement (il faut ≥2 aliments reconnus)."}
        (a1_name, g1, r1, p1, k1, _), (a2_name, g2, r2, p2, k2, _) = picked
        adjustables = [
            AdjustableItem(name=a1_name, min_g=0.0, max_g=None),
            AdjustableItem(name=a2_name, min_g=0.0, max_g=None),
        ]
    else:
        adjustables = req.adjustables

    # 4) Construire vecteur de départ (grammes initiaux des ajustables)
    names = []
    start = []
    bounds = []
    pk = []  # (p_per_g, k_per_g)
    for ad in adjustables:
        nm = best_match(ad.name)
        if not nm:
            return {"error": f"Aliment non trouvé: {ad.name}"}
        names.append(nm)
        # grammes actuels dans target (0 si absent)
        g0 = 0.0
        for it in tgt_list:
            if best_match(it["aliment"]) == nm:
                g0 = float(it["grams"]); break
        r = row_for(nm)
        if r is None:
            return {"error": f"Aliment non trouvé dans la base: {nm}"}
        p, k = _p_k_per_g(r)
        pk.append((p, k))
        lo = float(ad.min_g)
        hi = None if ad.max_g is None else float(ad.max_g)
        # si g0 < min_g, on part au min
        g0 = max(g0, lo)
        start.append(g0)
        bounds.append((lo, hi))

    start = np.array(start, dtype=float)
    P = np.array([p for p, _ in pk], dtype=float)
    K = np.array([k for _, k in pk], dtype=float)

    # 5) Construire le système A x = b sur les "delta grams"
    # A: 2 x n, b: 2 x 1
    A = np.vstack([P, K])         # 2 x n
    b = np.array([dp, dk])        # 2

    # On fait des itérations: solve -> appliquer bornes -> retirer vars saturées -> re-solve
    active = np.ones(len(start), dtype=bool)
    x = np.zeros_like(start)

    def solve_active(A, b, active_mask):
        A2 = A[:, active_mask]
        # cas n_active == 1: impossible d'égaliser 2 équations -> approximation (lstsq)
        if A2.shape[1] == 0:
            return None
        # moindres carrés
        sol, *_ = np.linalg.lstsq(A2.T, b, rcond=None)  # (n_active, ) en résolvant sur transpose
        return sol

    # boucle max 5 itérations
    for _ in range(5):
        if active.sum() == 0:
            break
        sol = solve_active(A, b, active)
        if sol is None:
            break
        x[active] = sol
        cand = start + x
        # appliquer bornes
        violated = np.zeros_like(active)
        for i, (lo, hi) in enumerate(bounds):
            if not active[i]: 
                continue
            if cand[i] < lo - 1e-9:
                cand[i] = lo
                violated[i] = True
            if hi is not None and cand[i] > hi + 1e-9:
                cand[i] = hi
                violated[i] = True
        if not violated.any():
            start = cand
            break
        # fixer les variables violées et ré-estimer pour les restantes
        for i in range(len(active)):
            if violated[i]:
                active[i] = False
                # Soustraire la contribution fixée de b
                fix_delta = cand[i] - start[i]
                b = b - np.array([P[i]*fix_delta, K[i]*fix_delta])
                start[i] = cand[i]
        x[:] = 0.0

    # 6) Arrondi puis micro-ajustement pour retomber pile
    step = max(1, int(req.round_to))
    rounded = np.array([round(g/step)*step for g in start], dtype=float)

    # recalcul du delta total obtenu
    delta_p = float(np.dot(P, rounded - np.array([g for g in rounded], dtype=float)))  # placeholder
    # Recalcule correctement: delta = rounded - (grammes initiaux AVANT ajustement)
    # => il faut re-construire grams initiaux "target"
    target_grams_map = {}
    for it in tgt_list:
        m = best_match(it["aliment"])
        target_grams_map[m] = float(it["grams"])
    deltas = []
    for i, nm in enumerate(names):
        g0 = target_grams_map.get(nm, 0.0)
        deltas.append(rounded[i] - g0)
    deltas = np.array(deltas, dtype=float)

    delta_p = float(np.dot(P, deltas))
    delta_k = float(np.dot(K, deltas))

    # résidus après arrondi
    rp = dp - delta_p
    rk = dk - delta_k

    # petit ajustement glouton: on choisit la variable au ratio le plus pertinent
    def try_nudge(var_idx, sign):
        new = rounded.copy()
        new[var_idx] = _clip(new[var_idx] + sign*step, bounds[var_idx][0], bounds[var_idx][1])
        return new

    # on essaye 30 coups max
    for _ in range(30):
        if abs(rp) < 0.01 and abs(rk) < 0.5:  # tolérances serrées
            break
        # priorité d'ajustement
        goal = "k" if req.priority == "kcal" else "p"
        # choisir la var qui modifie le plus la grandeur prioritaire
        idx = int(np.argmax(K if goal=="k" else P))
        # déterminer le signe à appliquer
        coeff = (K[idx] if goal=="k" else P[idx])
        if abs(coeff) < 1e-9:
            break
        sign = 1 if ((rk if goal=="k" else rp) > 0) else -1
        new = try_nudge(idx, sign)
        # recalcul rp, rk
        new_deltas = new - np.array([target_grams_map.get(nm, 0.0) for nm in names], dtype=float)
        new_rp = dp - float(np.dot(P, new_deltas))
        new_rk = dk - float(np.dot(K, new_deltas))
        # si on est mieux, on garde
        if abs(new_rp) + abs(new_rk) < abs(rp) + abs(rk):
            rounded = new
            rp, rk = new_rp, new_rk
        else:
            break

    # 7) Reconstruire le plan ajusté
    adjusted = []
    used = set()
    # réinjecte grammes arrondis pour les ajustables
    for it in tgt_list:
        m = best_match(it["aliment"])
        if m in names and m not in used:
            i = names.index(m)
            adjusted.append({"aliment": m, "grams": float(rounded[i])})
            used.add(m)
        else:
            adjusted.append({"aliment": it["aliment"], "grams": it["grams"]})
    # si un adjustable n'était pas présent avant, on l'ajoute
    for i, nm in enumerate(names):
        if nm not in [best_match(x["aliment"]) for x in tgt_list]:
            adjusted.append({"aliment": nm, "grams": float(rounded[i])})

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
        "adjusted_items": [{"name": n, "final_grams": float(rounded[i]), "bounds": bounds[i]} for i, n in enumerate(names)],
        "adjusted_plan": adjusted,
        "final_totals": final_tot,
        "residual_diff": diffs
    }
# ---------- SUGGESTION DE RECETTE (iso-kcal & iso-prot) ----------
from pydantic import BaseModel
import numpy as np

class SuggestReq(BaseModel):
    current: list[dict]          # [{"aliment": "...", "grams": 100}, ...]
    likes: list[str]             # ["chips","yogourt grec","fruit",...]
    round_to: int = 5

LIKE_MAP = {
    "chips": ["Croustilles", "Croustilles de maïs", "Chips"],
    "yogourt": ["Yogourt grec", "Yogourt, nature"],
    "yogourt grec": ["Yogourt grec, nature, 0%", "Yogourt grec, nature, 2%"],
    "fruit": ["Pomme", "Banane", "Fraises", "Bleuets", "Raisin"],
    "pâtes": ["Spaghetti, cuit", "Macaroni, cuit", "Pâtes, cuites"],
    "riz": ["Riz brun, grains longs, cuit", "Riz blanc à grains longs, cuit", "Riz sauvage, cuit"],
}

def _row_strict(name: str):
    rows = db[db["Aliment"] == name]
    return rows.iloc[0] if not rows.empty else None

def _per_g(row):
    return float(row["protein_g_per_100g"])/100.0, float(row["kcal_per_100g"])/100.0

def _best_like_matches(likes, foods, k=3):
    got = []
    low = [f.lower() for f in foods]
    for like in likes:
        keys = LIKE_MAP.get(like.strip().lower(), [like])
        for key in keys:
            keyl = key.lower()
            hits = [foods[i] for i, f in enumerate(low) if keyl in f]
            for h in hits:
                if h not in got: got.append(h)
            if len(got) >= 6: break
        if len(got) >= 6: break
    if not got:
        got = ["Yogourt grec, nature, 0%", "Pomme", "Spaghetti, cuit"]
    return got[:3]

@app.post("/suggest-plan")
def suggest_plan(req: SuggestReq):
    # totaux du plan actuel
    cur_tot = totals_for_plan(req.current)  # utilise les fonctions déjà présentes

    # candidats depuis les envies
    cand = _best_like_matches(req.likes, FOODS, k=3)
    P,K,names = [],[],[]
    for nm in cand:
        r = _row_strict(nm) or row_for(nm)
        if r is None: continue
        p,k = _per_g(r)
        if p==0 and k==0: continue
        names.append(nm); P.append(p); K.append(k)
    if len(P) < 2:
        return {"error":"Pas assez de candidats valides depuis les envies."}

    # Résoudre A x = b (iso-prot & iso-kcal), avec bornes simples
    A = np.vstack([P, K])                                  # 2 x n
    b = np.array([cur_tot["prot_g"], cur_tot["kcal"]])     # 2
    # bornes simples (éviter solutions absurdes)
    def bounds_for(nm):
        l = nm.lower()
        if "croustille" in l or "chips" in l: return (15, 60)
        if "yogourt" in l: return (100, 300)
        if "pomme" in l or "banane" in l or "fraise" in l: return (80, 250)
        if "spaghetti" in l or "pâtes" in l: return (100, 250)
        if "riz" in l: return (100, 250)
        return (0, None)

    # point de départ = bornes basses
    x = np.array([bounds_for(n)[0] for n in names], dtype=float)
    b2 = b - np.array([np.dot(P, x), np.dot(K, x)])

    active = [i for i in range(len(names)) if True]
    for _ in range(6):
        if not active: break
        A2 = A[:, active]
        sol, *_ = np.linalg.lstsq(A2.T, b2, rcond=None)
        cand_x = x.copy()
        for j, i in enumerate(active):
            cand_x[i] = x[i] + sol[j]
        violated = []
        for i, nm in enumerate(names):
            lo, hi = bounds_for(nm)
            if cand_x[i] < lo: cand_x[i] = lo; violated.append(i)
            if hi is not None and cand_x[i] > hi: cand_x[i] = hi; violated.append(i)
        if not violated:
            x = cand_x
            break
        # fige les violés et ré-absorbe dans b2
        for i in set(violated):
            dx = cand_x[i] - x[i]
            b2 = b2 - np.array([P[i]*dx, K[i]*dx])
            x[i] = cand_x[i]
            if i in active: active.remove(i)

    # arrondi et contrôle tolérances (±10 kcal, ±3 g prot)
    step = max(1, int(req.round_to))
    x = np.array([round(g/step)*step for g in x], dtype=float)

    suggestion = [{"aliment": names[i], "grams": float(x[i])} for i in range(len(names))]
    final = totals_for_plan(suggestion)
    diff = {"kcal": round(cur_tot["kcal"]-final["kcal"],2),
            "prot_g": round(cur_tot["prot_g"]-final["prot_g"],2)}

    # petit ajustement glouton si hors tolérances
    for _ in range(20):
        okK = abs(diff["kcal"]) <= 10
        okP = abs(diff["prot_g"]) <= 3
        if okK and okP: break
        # bouge l'item qui influence le plus la grandeur dominante
        goal = "kcal" if abs(diff["kcal"])>=abs(diff["prot_g"]) else "prot"
        idx = int(np.argmax(K if goal=="kcal" else P))
        sgn = 1 if (diff["kcal"]>0 if goal=="kcal" else diff["prot_g"]>0) else -1
        lo, hi = bounds_for(names[idx])
        newg = x[idx] + sgn*step
        if (hi is not None and newg>hi) or newg<lo: break
        x[idx] = newg
        suggestion[idx]["grams"] = float(newg)
        final = totals_for_plan(suggestion)
        diff = {"kcal": round(cur_tot["kcal"]-final["kcal"],2),
                "prot_g": round(cur_tot["prot_g"]-final["prot_g"],2)}

    return {
        "current_totals": cur_tot,
        "likes": req.likes,
        "candidates": names,
        "suggested_recipe": suggestion,
        "final_totals": final,
        "residual_diff": diff
    }
