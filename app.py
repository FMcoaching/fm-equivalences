# app.py — version "likes only" améliorée (précision accrue)

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
        from fm_equivalences import load_food_db, equivalent_portion
        DB = load_food_db("nutrient_values_clean.csv")
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
    def safe(col):
        try:
            v = float(row[col])
            return 0.0 if v != v else v
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
    mode: str = "carbs"

@app.post("/equivalence")
def equivalence(req: EqReq):
    lazy_load()
    if LOAD_ERR:
        raise HTTPException(status_code=500, detail=f"DB load failed: {LOAD_ERR}")
    try:
        res = equivalent_portion(db, req.source, req.grams, req.target, mode=req.mode)
        res["target_grams"] = round(res["target_grams"] / 5) * 5
        return res
    except Exception as e:
        raise HTTPException(status_code=422, detail={"error": str(e)})

# ---------- LIKE_MAP ----------
LIKE_MAP = {
    "chips": ["Croustilles", "Croustilles de maïs", "Chips"],
    "yogourt grec": ["Yogourt grec, nature, 0%", "Yogourt grec, nature, 2%"],
    "fruit": ["Pomme", "Banane", "Fraises", "Bleuets", "Raisin", "Orange"],
    "pâtes": ["Spaghetti, cuit", "Macaroni, cuit", "Pâtes, cuites"],
    "riz": ["Riz brun, grains longs, cuit", "Riz blanc à grains longs, cuit", "Riz sauvage, cuit"],
    "boeuf": ["Boeuf haché, extra maigre, émiétté, sauté", "Boeuf haché, extra maigre, cru"],
    "patate": ["Pomme de terre au four, chair et pelure", "Pomme de terre, bouillie, chair et pelure"],
    "haricot": ["Haricots noirs, en conserve, égouttés", "Haricots rouges, bouillis"],
    "jus de pomme": ["Jus de pomme, non sucré"],
}

def bounds_for(nm: str) -> Tuple[float, Optional[float]]:
    l = (nm or "").lower()
    if any(k in l for k in ["yogourt", "boeuf", "poulet", "dinde", "poisson", "tofu"]):
        return (90.0, 300.0)
    if any(k in l for k in ["pomme de terre", "spaghetti", "pâtes", "riz", "pain", "avoine"]):
        return (90.0, 250.0)
    if any(k in l for k in ["croustille", "chips", "huile", "beurre", "arachide"]):
        return (10.0, 60.0)
    return (50.0, 250.0)

def map_likes_to_exact_candidates(likes: List[str], foods: List[str]) -> List[str]:
    results: List[str] = []
    low = [f.lower() for f in foods]
    for like in likes:
        base = like.strip().lower()
        keys = LIKE_MAP.get(base, [like])
        picked = None
        for key in keys:
            k = key.lower().strip()
            contains = [foods[i] for i, nm in enumerate(low) if k in nm]
            if contains:
                picked = contains[0]; break
            close = difflib.get_close_matches(k, low, n=1, cutoff=0.6)
            if close:
                picked = foods[low.index(close[0])]; break
        if picked and picked not in results:
            results.append(picked)
    return results

class SuggestReq(BaseModel):
    current: List[Dict[str, Any]]
    likes:   List[str]
    round_to: int = 5

@app.post("/suggest-plan")
def suggest_plan(req: SuggestReq):
    lazy_load()
    if LOAD_ERR:
        raise HTTPException(status_code=500, detail=f"DB load failed: {LOAD_ERR}")

    try:
        if not req.current:
            raise HTTPException(status_code=422, detail={"error": "Aucun aliment fourni."})
        cur_tot = totals_for_plan(req.current)

        if not req.likes:
            raise HTTPException(status_code=422, detail={"error": "Aucune 'Envie' fournie."})
        cand = map_likes_to_exact_candidates(req.likes, FOODS)

        names, P, K, C, F, B = [], [], [], [], [], []
        for nm in cand:
            r = row_for(nm)
            if r is None: continue
            p_per_g, k_per_g = per_gram(r)
            if abs(p_per_g)<1e-9 and abs(k_per_g)<1e-9: continue
            names.append(nm)
            P.append(p_per_g); K.append(k_per_g)
            C.append(float(r.get("carb_g_per_100g", 0) or 0)/100.0)
            F.append(float(r.get("fat_g_per_100g", 0) or 0)/100.0)
            B.append(bounds_for(nm))

        if len(P)<2:
            raise HTTPException(status_code=422, detail={"error":"Pas assez de candidats."})

        # --- Ajustement exact 2x2 (prot/kcal) ---
        x0 = np.array([B[i][0] for i in range(len(names))], dtype=float)
        restP, restK = np.dot(P,x0), np.dot(K,x0)
        need = np.array([cur_tot["prot_g"]-restP, cur_tot["kcal"]-restK])
        i1 = int(np.argmax(np.abs(P)))
        alpha = (np.dot(P,K)/(np.dot(P,P)+1e-9))
        residual_vec = np.array(K)-alpha*np.array(P)
        i2 = int(np.argmax(np.abs(residual_vec)))
        if i2==i1: i2=(i1+1)%len(names)
        A2 = np.array([[P[i1],P[i2]],[K[i1],K[i2]]])
        try:
            sol2 = np.linalg.solve(A2,need)
            x_exact = x0.copy(); x_exact[i1]+=sol2[0]; x_exact[i2]+=sol2[1]
            x = x_exact
        except Exception: x=x0

        # --- Arrondi d’affichage mais raffinement fin ---
        step_user = max(1,int(req.round_to))
        x = np.array([round(g/step_user)*step_user for g in x],dtype=float)

        suggestion=[{"aliment":names[i],"grams":float(x[i])} for i in range(len(names))]
        final=totals_for_plan(suggestion)
        diff={k:round(cur_tot[k]-final[k],2) for k in ["kcal","prot_g","carb_g","fat_g"]}

        # --- Raffinement précis (1 g) ---
        for _ in range(200):
            okK=abs(diff["kcal"])<=10; okP=abs(diff["prot_g"])<=3
            okC=abs(diff["carb_g"])<=5; okF=abs(diff["fat_g"])<=5
            if okK and okP and okC and okF: break

            residuals={"kcal":abs(diff["kcal"])/10,"prot_g":abs(diff["prot_g"])/3,
                       "carb_g":abs(diff["carb_g"])/5,"fat_g":abs(diff["fat_g"])/5}
            key=max(residuals,key=residuals.get)
            coeff_map={"kcal":K,"prot_g":P,"carb_g":C,"fat_g":F}
            coeffs=coeff_map[key]; move=1 if diff[key]>0 else -1

            best=None; gain=-1
            for i in range(len(names)):
                lo,hi=B[i]; nxt=x[i]+move
                if (hi is not None and nxt>hi) or nxt<lo: continue
                g=abs(coeffs[i])
                if g>gain: gain=g; best=i
            if best is None: break

            x[best]+=move; suggestion[best]["grams"]=float(x[best])
            final=totals_for_plan(suggestion)
            diff={k:round(cur_tot[k]-final[k],2) for k in ["kcal","prot_g","carb_g","fat_g"]}

        return {"current_totals":cur_tot,"likes":req.likes,"candidates":names,
                "suggested_recipe":suggestion,"final_totals":final,"residual_diff":diff}

    except HTTPException: raise
    except Exception as e:
        raise HTTPException(status_code=422, detail={"error":f"suggest-plan failed: {e}"})

@app.post("/upload-plan")
def upload_plan(file: UploadFile=File(...)):
    return {"filename":file.filename,"note":"OCR non implémenté encore."}
