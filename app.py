# app.py — version complète, robuste

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
import difflib
import numpy as np

# --------- Lazy load DB + fonctions existantes ----------
db = None
FOODS: List[str] = []
LOAD_ERR: Optional[str] = None

def lazy_load():
    """Charge la base et les utilitaires une seule fois, sans planter au démarrage."""
    global db, FOODS, LOAD_ERR, equivalent_portion, load_food_db
    if db is not None or LOAD_ERR is not None:
        return
    try:
        from fm_equivalences import load_food_db, equivalent_portion  # tes fonctions
        DB = load_food_db("nutrient_values_clean.csv")  # fichier à la racine
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
    """Retourne la ligne (pandas Series) pour un aliment par son nom exact/rapproché."""
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
            return float(row[col])
        except Exception:
            return 0.0
    return {
        "kcal":  round(safe("kcal_per_100g")        * f, 4),
        "prot_g":round(safe("protein_g_per_100g")   * f, 4),
        "carb_g":round(safe("carb_g_per_100g")      * f, 4),
        "fat_g": round(safe("fat_g_per_100g")       * f, 4),
        "fiber_g":round(safe("fiber_g_per_100g")    * f, 4),
    }

def totals_for_plan(items: List[Dict[str, Any]]) -> Dict[str, float]:
    t = {"kcal": 0.0, "prot_g": 0.0, "carb_g": 0.0, "fat_g": 0.0, "fiber_g": 0.0}
    for it in items:
        r = row_for(it.get("aliment", ""))
        if r is None: 
            # on ignore ce qu'on ne trouve pas
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
            v = float(row[col]); 
            return 0.0 if v != v else v
        except Exception:
            return 0.0
    return safe("protein_g_per_100g")/100.0, safe("kcal_per_100g")/100.0

# --------- App & static ----------
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

# --------- Search (auto-complétion) ----------
@app.get("/search")
def search(q: str):
    lazy_load()
    if LOAD_ERR:
        raise HTTPException(status_code=500, detail=f"DB load failed: {LOAD_ERR}")
    hits = [x for x in FOODS if q.lower() in x.lower()][:8]
    if not hits:
        hits = difflib.get_close_matches(q, FOODS, n=8, cutoff=0.5)
    return {"results": hits}

# --------- Équivalence simple (déjà existante) ----------
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

# ---------- SUGGESTION DE RECETTE (iso-kcal & iso-prot) ----------
class SuggestReq(BaseModel):
    current: List[Dict[str, Any]]   # [{"aliment": "...", "grams": 100}, ...]
    likes:   List[str]              # ["chips","yogourt grec","fruit",...]
    round_to: int = 5

LIKE_MAP = {
    "chips": ["Croustilles", "Croustilles de maïs", "Chips"],
    "yogourt": ["Yogourt grec", "Yogourt, nature"],
    "yogourt grec": ["Yogourt grec, nature, 0%", "Yogourt grec, nature, 2%"],
    "fruit": ["Pomme", "Banane", "Fraises", "Bleuets", "Raisin"],
    "pâtes": ["Spaghetti, cuit", "Macaroni, cuit", "Pâtes, cuites"],
    "riz": ["Riz brun, grains longs, cuit", "Riz blanc à grains longs, cuit", "Riz sauvage, cuit"],
    "croustilles": ["Croustilles", "Croustilles de maïs"],
}

def best_like_matches(likes: List[str], foods: List[str], k: int = 3) -> List[str]:
    got: List[str] = []
    low = [f.lower() for f in foods]
    for like in likes:
        base = like.strip().lower()
        keys = LIKE_MAP.get(base, [like])
        # 1) contains
        for key in keys:
            keyl = key.lower()
            hits = [foods[i] for i, f in enumerate(low) if keyl in f]
            for h in hits:
                if h not in got:
                    got.append(h)
            if len(got) >= 6:
                break
        if len(got) >= 6:
            break
        # 2) fuzzy
        close = difflib.get_close_matches(base, low, n=4, cutoff=0.6)
        for c in close:
            idx = low.index(c)
            cand = foods[idx]
            if cand not in got:
                got.append(cand)
        if len(got) >= 6:
            break
    if not got:
        got = ["Yogourt grec, nature, 0%", "Pomme", "Spaghetti, cuit"]
    # dédoublonnage + limite
    out: List[str] = []
    for x in got:
        if x not in out:
            out.append(x)
    return out[:k]

def bounds_for(nm: str) -> Tuple[float, Optional[float]]:
    l = nm.lower()
    if "croustille" in l or "chips" in l: return (15, 60)
    if "yogourt" in l: return (100, 300)
    if "pomme" in l or "banane" in l or "fraise" in l: return (80, 250)
    if "spaghetti" in l or "pâtes" in l: return (100, 250)
    if "riz" in l: return (100, 250)
    return (0, None)

@app.post("/suggest-plan")
def suggest_plan(req: SuggestReq):
    lazy_load()
    if LOAD_ERR:
        raise HTTPException(status_code=500, detail=f"DB load failed: {LOAD_ERR}")

    try:
        # 1) totaux actuels
        cur_tot = totals_for_plan(req.current)

        # 2) candidats à partir des envies
        cand = best_like_matches(req.likes, FOODS, k=3)

        names: List[str] = []
        P: List[float] = []
        K: List[float] = []
        bounds: List[Tuple[float, Optional[float]]] = []

        for nm in cand:
            r = row_for(nm)  # (fix: pas de "or" sur Series)
            if r is None:
                continue
            p, k = per_gram(r)
            if abs(p) < 1e-9 and abs(k) < 1e-9:
                continue
            names.append(nm)
            P.append(p)
            K.append(k)
            bounds.append(bounds_for(nm))

        if len(P) < 2:
            raise HTTPException(
                status_code=422,
                detail={"error": "Pas assez de candidats valides depuis les envies.", "candidates_seen": cand}
            )

        # 3) Résoudre A x = b (iso-prot & iso-kcal) avec bornes
        A = np.vstack([P, K])                                # 2 x n
        b = np.array([cur_tot["prot_g"], cur_tot["kcal"]])   # 2

        # point de départ = bornes basses
        x = np.array([bounds[i][0] if bounds[i][0] is not None else 0.0
                      for i in range(len(names))], dtype=float)
        b2 = b - np.array([np.dot(P, x), np.dot(K, x)])

        active = [i for i in range(len(names))]
        for _ in range(6):
            if not active:
                break
            A2 = A[:, active]
            # moindres carrés
            sol, *_ = np.linalg.lstsq(A2, b2, rcond=None)
            cand_x = x.copy()
            for j, i in enumerate(active):
                cand_x[i] = x[i] + float(sol[j])
            # bornes
            violated = []
            for i in range(len(names)):
                lo, hi = bounds[i]
                if cand_x[i] < lo:
                    cand_x[i] = lo; violated.append(i)
                if hi is not None and cand_x[i] > hi:
                    cand_x[i] = hi; violated.append(i)
            if not violated:
                x = cand_x
                break
            # fige violés et soustrait contribution
            for i in set(violated):
                dx = cand_x[i] - x[i]
                b2 = b2 - np.array([P[i]*dx, K[i]*dx])
                x[i] = cand_x[i]
                if i in active:
                    active.remove(i)

        # 4) Arrondi + micro-ajustements (±10 kcal / ±3 g prot)
        step = max(1, int(req.round_to))
        x = np.array([round(g/step)*step for g in x], dtype=float)

        suggestion = [{"aliment": names[i], "grams": float(x[i])} for i in range(len(names))]
        final = totals_for_plan(suggestion)
        diff = {"kcal": round(cur_tot["kcal"]-final["kcal"], 2),
                "prot_g": round(cur_tot["prot_g"]-final["prot_g"], 2)}

        for _ in range(30):
            okK = abs(diff["kcal"]) <= 10
            okP = abs(diff["prot_g"]) <= 3
            if okK and okP:
                break
            # choisir la variable la plus efficace pour la grandeur dominante
            goal_k = abs(diff["kcal"]) >= abs(diff["prot_g"])
            coeffs = K if goal_k else P
            idx = int(np.argmax(np.abs(coeffs)))
            lo, hi = bounds[idx]
            move = step if (diff["kcal"] > 0 if goal_k else diff["prot_g"] > 0) else -step
            newg = x[idx] + move
            if (hi is not None and newg > hi) or newg < lo:
                newg = x[idx] - move
                if (hi is not None and newg > hi) or newg < lo:
                    break
            x[idx] = newg
            suggestion[idx]["grams"] = float(newg)
            final = totals_for_plan(suggestion)
            diff = {"kcal": round(cur_tot["kcal"]-final["kcal"], 2),
                    "prot_g": round(cur_tot["prot_g"]-final["prot_g"], 2)}

        return {
            "current_totals": cur_tot,
            "likes": req.likes,
            "candidates": names,
            "suggested_recipe": suggestion,
            "final_totals": final,
            "residual_diff": diff
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail={"error": f"suggest-plan failed: {str(e)}"})

# ---------- (optionnel) Upload image placeholder ----------
@app.post("/upload-plan")
def upload_plan(file: UploadFile = File(...)):
    # Placeholder pour l’instant : retourne juste les meta
    return {"filename": file.filename, "note": "OCR non implémenté encore."}
