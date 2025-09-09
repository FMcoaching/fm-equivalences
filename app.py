# app.py — version "likes only" (Python 3.9 compatible)

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import difflib

# --------- Lazy load DB + utilitaires ---------
db = None
FOODS: List[str] = []
LOAD_ERR: Optional[str] = None

def lazy_load():
    """Charge la base une seule fois."""
    global db, FOODS, LOAD_ERR, equivalent_portion, load_food_db
    if db is not None or LOAD_ERR is not None:
        return
    try:
        from fm_equivalences import load_food_db, equivalent_portion  # tes fonctions
        DB = load_food_db("nutrient_values_clean.csv")  # CSV à la racine du repo
        foods = sorted(DB["Aliment"].dropna().unique().tolist())
        db = DB
        FOODS = foods
        globals()["equivalent_portion"] = equivalent_portion
        globals()["load_food_db"] = load_food_db
    except Exception as e:
        LOAD_ERR = str(e)

def best_match(name: str) -> Optional[str]:
    lazy_load()
    if not name or not FOODS:
        return None
    contains = [x for x in FOODS if name.lower() in x.lower()]
    if contains:
        return contains[0]
    close = difflib.get_close_matches(name, FOODS, n=1, cutoff=0.5)
    return close[0] if close else None

def row_for(name: str):
    """Retourne la ligne (pandas Series) correspondant au meilleur libellé."""
    lazy_load()
    if LOAD_ERR or db is None:
        return None
    m = best_match(name)
    if not m:
        return None
    rows = db[db["Aliment"] == m]
    return rows.iloc[0] if not rows.empty else None

def macros_for_grams(row, grams: float) -> Dict[str, float]:
    f = grams / 100.0
    def safe(col):
        try:
            v = float(row[col])
            return v
        except Exception:
            return 0.0
    return {
        "kcal": round(safe("kcal_per_100g") * f, 4),
        "prot_g": round(safe("protein_g_per_100g") * f, 4),
        "carb_g": round(safe("carb_g_per_100g") * f, 4),
        "fat_g": round(safe("fat_g_per_100g") * f, 4),
        "fiber_g": round(safe("fiber_g_per_100g") * f, 4),
    }

def totals_for_plan(items: List[Dict[str, Any]]) -> Dict[str, float]:
    t = {"kcal": 0.0, "prot_g": 0.0, "carb_g": 0.0, "fat_g": 0.0, "fiber_g": 0.0}
    for it in items:
        r = row_for(it.get("aliment", ""))
        if r is None: 
            continue
        g = float(it.get("grams", 0) or 0)
        m = macros_for_grams(r, g)
        for k in t.keys():
            t[k] += m[k]
    for k in t.keys():
        t[k] = round(t[k], 2)
    return t

def per_gram(row) -> Tuple[float, float]:
    """Retourne (prot/g, kcal/g)."""
    def safe(col):
        try:
            v = float(row[col])
            return 0.0 if v != v else v  # gère NaN
        except Exception:
            return 0.0
    return safe("protein_g_per_100g")/100.0, safe("kcal_per_100g")/100.0

# --------- App & static ---------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")

@app.get("/health")
def health():
    lazy_load()
    if LOAD_ERR:
        return JSONResponse({"status": "degraded", "error": LOAD_ERR}, status_code=500)
    return {"status": "ok"}

# --------- Autocomplétion ---------
@app.get("/search")
def search(q: str):
    lazy_load()
    if LOAD_ERR:
        raise HTTPException(status_code=500, detail=f"DB load failed: {LOAD_ERR}")
    hits = [x for x in FOODS if q.lower() in x.lower()][:8]
    if not hits:
        hits = difflib.get_close_matches(q, FOODS, n=8, cutoff=0.5)
    return {"results": hits}

# --------- Équivalence simple ---------
class EqReq(BaseModel):
    source: str
    grams: float
    target: str
    mode: str = "carbs"  # "carbs" ou "kcal"

@app.post("/equivalence")
def equivalence(req: EqReq):
    lazy_load()
    if LOAD_ERR:
        raise HTTPException(status_code=500, detail=f"DB load failed: {LOAD_ERR}")
    try:
        res = equivalent_portion(db, req.source, req.grams, req.target, mode=req.mode)
        res["target_grams"] = round(res["target_grams"] / 5) * 5  # arrondi pratique
        return res
    except Exception as e:
        raise HTTPException(status_code=422, detail={"error": str(e)})

# ---------- LIKE_MAP (synonymes/variantes usuelles) ----------
LIKE_MAP = {
    "chips": ["Croustilles", "Croustilles de maïs", "Chips"],
    "yogourt": ["Yogourt grec", "Yogourt, nature"],
    "yogourt grec": ["Yogourt grec, nature, 0%", "Yogourt grec, nature, 2%"],
    "fruit": ["Pomme", "Banane", "Fraises", "Bleuets", "Raisin", "Orange"],
    "pâtes": ["Spaghetti, cuit", "Macaroni, cuit", "Pâtes, cuites"],
    "riz": ["Riz brun, grains longs, cuit", "Riz blanc à grains longs, cuit", "Riz sauvage, cuit"],
    "croustilles": ["Croustilles", "Croustilles de maïs"],
    "boeuf": ["Boeuf haché, extra maigre, émiétté, sauté", "Boeuf haché, extra maigre, cru", "Bifteck de boeuf, grillé"],
    "boeuf hache": ["Boeuf haché, extra maigre, émiétté, sauté", "Boeuf haché, extra maigre, cru"],
    "patate": ["Pomme de terre au four, chair et pelure", "Pomme de terre, bouillie, chair et pelure"],
    "pomme de terre": ["Pomme de terre au four, chair et pelure", "Pomme de terre, bouillie, chair et pelure", "Farine de pomme de terre"],
    "haricot": ["Haricots noirs, en conserve, égouttés", "Haricots rouges, bouillis", "Haricots blancs, bouillis"],
    "jus de pomme": ["Jus de pomme, non sucré"],
}

# ---------- Bornes par aliment (plausibles) ----------
def bounds_for(nm: str) -> Tuple[float, Optional[float]]:
    # seuils simples basés sur le type d'aliment
    l = (nm or "").lower()
    if "yogourt" in l or "boeuf" in l or "poulet" in l or "dinde" in l or "poisson" in l or "tofu" in l:
        return (90.0, 300.0)     # protéines
    if "pomme de terre" in l or "spaghetti" in l or "pâtes" in l or "riz" in l or "pain" in l or "avoine" in l:
        return (90.0, 250.0)     # glucides
    if "croustille" in l or "chips" in l or "huile" in l or "beurre" in l or "arachide" in l:
        return (10.0, 60.0)      # gras/snacks denses
    return (50.0, 250.0)         # par défaut

# ---------- Mapping strict "likes → candidats" ----------
def map_likes_to_exact_candidates(likes: List[str], foods: List[str]) -> List[str]:
    """Retourne 1 candidat par like (dans l'ordre), sans ajouter d'aliments non demandés.
       - essaie d'abord LIKE_MAP (synonymes/variantes),
       - sinon 'contains',
       - sinon fuzzy (rapproché),
       - si introuvable => ignoré (on validera ensuite).
    """
    results: List[str] = []
    low = [f.lower() for f in foods]
    for like in likes:
        base = like.strip().lower()
        keys = LIKE_MAP.get(base, [like])
        picked = None
        for key in keys:
            k = key.lower().strip()
            # contains
            contains = [foods[i] for i, nm in enumerate(low) if k in nm]
            if contains:
                picked = contains[0]; break
            # fuzzy
            close = difflib.get_close_matches(k, low, n=1, cutoff=0.6)
            if close:
                picked = foods[low.index(close[0])]; break
        if picked and picked not in results:
            results.append(picked)
        # sinon: on ignore ce like (on signalera plus bas)
    return results

# ---------- Suggestion de recette (UTILISE UNIQUEMENT les likes) ----------
class SuggestReq(BaseModel):
    # 'current' = uniquement les aliments à remplacer (ce que le client a entré)
    current: List[Dict[str, Any]]
    likes:   List[str]          # ce qu’il veut manger à la place (2+ éléments)
    round_to: int = 5

@app.post("/suggest-plan")
def suggest_plan(req: SuggestReq):
    lazy_load()
    if LOAD_ERR:
        raise HTTPException(status_code=500, detail=f"DB load failed: {LOAD_ERR}")

    try:
        # 1) Totaux de la PARTIE à remplacer (les items fournis)
        if not req.current:
            raise HTTPException(status_code=422, detail={"error": "Aucun aliment fourni dans 'current'."})
        cur_tot = totals_for_plan(req.current)  # cibles ±10 kcal / ±3 g prot

        # 2) Construire les candidats STRICTEMENT à partir des "likes"
        if not req.likes:
            raise HTTPException(status_code=422, detail={"error": "Aucune 'Envie' fournie."})

        cand = map_likes_to_exact_candidates(req.likes, FOODS)

        # validations
        if len(cand) == 0:
            raise HTTPException(status_code=422, detail={
                "error": "Aucun des éléments saisis dans 'Envies' n'a été reconnu dans la base.",
                "hint": "Essaye des libellés proches de l’autocomplétion (ex.: 'boeuf haché extra maigre', 'pomme de terre, bouillie')."
            })
        if len(cand) < 2:
            raise HTTPException(status_code=422, detail={
                "error": "Pour calculer une alternative, saisis au moins deux ingrédients dans 'Envies'.",
                "hint": "Ex.: 'boeuf, patate' ou 'yogourt grec, fruit'."
            })
        # au moins une source protéique (évite 100% glucides)
        has_protein = False
        for nm in cand:
            r = row_for(nm)
            if r is not None:
                p100 = float(r.get("protein_g_per_100g", 0) or 0)
                if p100 >= 10:
                    has_protein = True; break
        if not has_protein:
            raise HTTPException(status_code=422, detail={
                "error": "Les 'Envies' ne contiennent aucune source de protéines.",
                "hint": "Ajoute une protéine (ex.: 'yogourt grec', 'poulet', 'boeuf haché extra maigre')."
            })

        # 3) Prépare le système A x = b
       names: List[str] = []
P: List[float] = []   # prot/g
K: List[float] = []   # kcal/g
C: List[float] = []   # carb/g   <-- AJOUT
F: List[float] = []   # fat/g    <-- AJOUT
B: List[Tuple[float, Optional[float]]] = []


        for nm in cand:
            r = row_for(nm)
            if r is None:
                continue
            p_per_g, k_per_g = per_gram(r)
            if abs(p_per_g) < 1e-9 and abs(k_per_g) < 1e-9:
                continue
            names.append(nm)
            P.append(p_per_g)
            K.append(k_per_g)
            c_per_g = float(r.get("carb_g_per_100g", 0) or 0) / 100.0
f_per_g = float(r.get("fat_g_per_100g", 0) or 0) / 100.0
C.append(c_per_g)
F.append(f_per_g)
            B.append(bounds_for(nm))

        if len(P) < 2:
            raise HTTPException(status_code=422, detail={
                "error": "Pas assez de candidats valides depuis les envies.",
                "candidates_seen": cand
            })

        A = np.vstack([P, K])                                # 2 x n
        b = np.array([cur_tot["prot_g"], cur_tot["kcal"]])   # 2

        # 4) Résolution avec bornes simples
        x = np.array([B[i][0] if B[i][0] is not None else 0.0 for i in range(len(names))], dtype=float)
        b2 = b - np.array([np.dot(P, x), np.dot(K, x)])

        active = [i for i in range(len(names))]
        for _ in range(6):
            if not active:
                break
            A2 = A[:, active]
            sol, *_ = np.linalg.lstsq(A2, b2, rcond=None)  # *** pas de transpose ***
            cand_x = x.copy()
            for j, i in enumerate(active):
                cand_x[i] = x[i] + float(sol[j])
            violated = []
            for i in range(len(names)):
                lo, hi = B[i]
                if cand_x[i] < lo:
                    cand_x[i] = lo; violated.append(i)
                if hi is not None and cand_x[i] > hi:
                    cand_x[i] = hi; violated.append(i)
            if not violated:
                x = cand_x
                break
            # fige violés et soustrait leur contribution
            for i in set(violated):
                dx = cand_x[i] - x[i]
                b2 = b2 - np.array([P[i]*dx, K[i]*dx])
                x[i] = cand_x[i]
                if i in active:
                    active.remove(i)

        # 5) Arrondi + micro-ajustements (±10 kcal / ±3 g prot)
        step = max(1, int(req.round_to))
        x = np.array([round(g/step)*step for g in x], dtype=float)

        suggestion = [{"aliment": names[i], "grams": float(x[i])} for i in range(len(names))]
        final = totals_for_plan(suggestion)
       diff = {
    "kcal":   round(cur_tot["kcal"]   - final["kcal"],   2),
    "prot_g": round(cur_tot["prot_g"] - final["prot_g"], 2),
    "carb_g": round(cur_tot["carb_g"] - final["carb_g"], 2),
    "fat_g":  round(cur_tot["fat_g"]  - final["fat_g"],  2),
}

        # petite boucle de raffinement si hors tolérances
        for _ in range(30):
           okK = abs(diff["kcal"]) <= 10
okP = abs(diff["prot_g"]) <= 3
okC = abs(cur_tot["carb_g"] - final["carb_g"]) <= 5
okF = abs(cur_tot["fat_g"] - final["fat_g"]) <= 5
if okK and okP and okC and okF:
    break

 # Choisir la dimension avec l'écart relatif le plus grand
residuals = {
    "kcal":   abs(diff["kcal"])   / 10.0,  # normalise par la tolérance
    "prot_g": abs(diff["prot_g"]) / 3.0,
    "carb_g": abs(diff["carb_g"]) / 5.0,
    "fat_g":  abs(diff["fat_g"])  / 5.0,
}
# clé avec la plus grande violation
key = max(residuals, key=residuals.get)

# Coefficients par gramme associés à la dimension choisie
coeff_map = {
    "kcal":   K,
    "prot_g": P,
    "carb_g": C,
    "fat_g":  F,
}
coeffs = coeff_map[key]

# Signe du mouvement (réduit l'écart dans la dimension 'key')
need_positive = (diff[key] > 0)   # il faut augmenter la dimension si diff>0, sinon diminuer
move = step if need_positive else -step

# Choisir l'ingrédient le plus "efficace" pour cette dimension
idx = int(np.argmax(np.abs(coeffs)))
lo, hi = B[idx]
newg = x[idx] + move

# Respect des bornes
if (hi is not None and newg > hi) or newg < lo:
    newg = x[idx] - move
    if (hi is not None and newg > hi) or newg < lo:
        break

# Appliquer
x[idx] = newg
suggestion[idx]["grams"] = float(newg)

# Recalculer les totaux & écarts
final = totals_for_plan(suggestion)
diff = {
    "kcal":   round(cur_tot["kcal"]   - final["kcal"],   2),
    "prot_g": round(cur_tot["prot_g"] - final["prot_g"], 2),
    "carb_g": round(cur_tot["carb_g"] - final["carb_g"], 2),
    "fat_g":  round(cur_tot["fat_g"]  - final["fat_g"],  2),

}


        return {
            "scope": "replace_only_these_items",
            "current_totals": cur_tot,
            "likes": req.likes,
            "candidates": names,              # correspond 1:1 aux likes reconnus
            "suggested_recipe": suggestion,   # UNIQUEMENT les likes (avec grammes)
            "final_totals": final,
            "residual_diff": diff
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail={"error": f"suggest-plan failed: {str(e)}"})

# ---------- Upload image (placeholder) ----------
@app.post("/upload-plan")
def upload_plan(file: UploadFile = File(...)):
    return {"filename": file.filename, "note": "OCR non implémenté encore."}
