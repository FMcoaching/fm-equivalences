# app.py — version complète, robuste (Python 3.9 compatible)

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
        from fm_equivalences import load_food_db, equivalent_portion  # fonctions de ton module
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

# ---------- Helpers LIKE & catégories ----------
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

def get_macros_100(row):
    def safe(col):
        try:
            v = float(row[col]); 
            return 0.0 if v != v else v
        except Exception:
            return 0.0
    return {"p": safe("protein_g_per_100g"),
            "c": safe("carb_g_per_100g"),
            "f": safe("fat_g_per_100g"),
            "k": safe("kcal_per_100g"),
            "fi": safe("fiber_g_per_100g")}

def classify_by_macros(row) -> str:
    m = get_macros_100(row)
    p, c, f = m["p"], m["c"], m["f"]
    if p >= 10 and p >= c and p >= f:   # dominant protéine
        return "protein"
    if f >= 10 and f > p and f >= c:    # très gras
        return "fat"
    if c >= 15 and c >= p and c >= f:   # dominant glucides
        return "carb"
    return "mixed"

def find_in_foods(keys: List[str], foods: List[str], limit: int = 6) -> List[str]:
    out: List[str] = []
    low = [f.lower() for f in foods]
    for key in keys:
        k = key.lower().strip()
        # contains
        hits = [foods[i] for i, nm in enumerate(low) if k in nm]
        for h in hits:
            if h not in out:
                out.append(h)
        # fuzzy
        close = difflib.get_close_matches(k, low, n=4, cutoff=0.6)
        for c in close:
            cand = foods[low.index(c)]
            if cand not in out:
                out.append(cand)
        if len(out) >= limit:
            break
    return out[:limit]

def choose_bucketed_candidates_from_likes(likes: List[str], foods: List[str], want: int = 3) -> List[str]:
    # 1) collecte brute
    raw: List[str] = []
    for like in likes:
        keys = LIKE_MAP.get(like.lower().strip(), [like])
        raw.extend(find_in_foods(keys, foods, limit=6))
    if not raw:
        raw = find_in_foods(["Yogourt grec", "Pomme", "Spaghetti, cuit"], foods, limit=6)

    # 2) classer
    buckets = {"protein": [], "carb": [], "fat": [], "mixed": []}
    seen = set()
    for nm in raw:
        if nm in seen:
            continue
        r = row_for(nm)
        if r is None:
            continue
        cat = classify_by_macros(r)
        buckets[cat].append(nm)
        seen.add(nm)

    # 3) garantir ≥1 protéine
    chosen: List[str] = []
    if buckets["protein"]:
        chosen.append(buckets["protein"][0])
    else:
        for fb in ["Yogourt grec, nature, 0%", "Poulet, poitrine, sans peau, rôti", "Dinde hachée, extra maigre, émiéttée, sautée"]:
            bm = best_match(fb)
            if bm:
                chosen.append(bm); break

    # 4) compléter avec carb puis mixed/fat/protein
    for cat in ["carb", "mixed", "fat", "protein"]:
        for nm in buckets[cat]:
            if nm not in chosen:
                chosen.append(nm)
            if len(chosen) >= want:
                break
        if len(chosen) >= want:
            break

    return chosen[:want]

def bounds_for(nm: str) -> Tuple[float, Optional[float]]:
    r = row_for(nm)
    cat = classify_by_macros(r) if r is not None else "mixed"
    if cat == "protein": return (90.0, 300.0)
    if cat == "carb":    return (90.0, 250.0)
    if cat == "fat":     return (10.0, 60.0)
    return (50.0, 250.0)  # mixed

# ---------- Suggestion de recette (multi-éléments fournis uniquement) ----------
class SuggestReq(BaseModel):
    # Ici, 'current' = uniquement les aliments que le client veut remplacer
    current: List[Dict[str, Any]]
    likes:   List[str]          # ce qu’il veut manger à la place
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

        # 2) Candidats depuis les envies (≥1 protéine garanti)
        cand = choose_bucketed_candidates_from_likes(req.likes, FOODS, want=3)

        names: List[str] = []
        P: List[float] = []
        K: List[float] = []
        B: List[Tuple[float, Optional[float]]] = []

        for nm in cand:
            r = row_for(nm)
            if r is None:
                continue
            p_per_g, k_per_g = per_gram(r)  # prot/g, kcal/g
            if abs(p_per_g) < 1e-9 and abs(k_per_g) < 1e-9:
                continue
            names.append(nm)
            P.append(p_per_g)
            K.append(k_per_g)
            B.append(bounds_for(nm))

        if len(P) < 2:
            raise HTTPException(status_code=422, detail={"error": "Pas assez de candidats valides depuis les envies.", "candidates_seen": cand})

        # 3) Résoudre A x = b (iso-prot & iso-kcal) avec bornes
        A = np.vstack([P, K])                                # 2 x n
        b = np.array([cur_tot["prot_g"], cur_tot["kcal"]])   # 2

        # point de départ = bornes basses
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
            # bornes
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
        diff = {"kcal": round(cur_tot["kcal"] - final["kcal"], 2),
                "prot_g": round(cur_tot["prot_g"] - final["prot_g"], 2)}

        # micro-ajustements si hors tolérances
        for _ in range(30):
            okK = abs(diff["kcal"]) <= 10
            okP = abs(diff["prot_g"]) <= 3
            if okK and okP:
                break
            # cible grandeur dominante
            goal_k = abs(diff["kcal"]) >= abs(diff["prot_g"])
            coeffs = K if goal_k else P
            idx = int(np.argmax(np.abs(coeffs)))
            lo, hi = B[idx]
            move = step if (diff["kcal"] > 0 if goal_k else diff["prot_g"] > 0) else -step
            newg = x[idx] + move
            if (hi is not None and newg > hi) or newg < lo:
                newg = x[idx] - move
                if (hi is not None and newg > hi) or newg < lo:
                    break
            x[idx] = newg
            suggestion[idx]["grams"] = float(newg)
            final = totals_for_plan(suggestion)
            diff = {"kcal": round(cur_tot["kcal"] - final["kcal"], 2),
                    "prot_g": round(cur_tot["prot_g"] - final["prot_g"], 2)}

        return {
            "scope": "replace_only_these_items",
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

# ---------- Upload image (placeholder) ----------
@app.post("/upload-plan")
def upload_plan(file: UploadFile = File(...)):
    return {"filename": file.filename, "note": "OCR non implémenté encore."}
