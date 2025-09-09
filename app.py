# app.py — "likes only" priorisé (prot -> carb -> fat), sans grammes négatifs

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
            v = float(row[col]); return v if v==v else 0.0
        except Exception:
            return 0.0
    return {
        "kcal":  round(safe("kcal_per_100g")       * f, 4),
        "prot_g":round(safe("protein_g_per_100g")  * f, 4),
        "carb_g":round(safe("carb_g_per_100g")     * f, 4),
        "fat_g": round(safe("fat_g_per_100g")      * f, 4),
        "fiber_g":round(safe("fiber_g_per_100g")   * f, 4),
    }

def totals_for_plan(items: List[Dict[str, Any]]) -> Dict[str, float]:
    t = {"kcal": 0.0, "prot_g": 0.0, "carb_g": 0.0, "fat_g": 0.0, "fiber_g": 0.0}
    for it in items:
        r = row_for(it.get("aliment", ""))
        if r is None: continue
        g = float(it.get("grams", 0) or 0)
        m = macros_for_grams(r, g)
        for k in t.keys(): t[k] += m[k]
    for k in t.keys(): t[k] = round(t[k], 2)
    return t

def per_gram(row) -> Dict[str, float]:
    def safe(col):
        try:
            v = float(row[col]); return v if v==v else 0.0
        except Exception:
            return 0.0
    return {
        "p": safe("protein_g_per_100g")/100.0,
        "c": safe("carb_g_per_100g")/100.0,
        "f": safe("fat_g_per_100g")/100.0,
        "k": safe("kcal_per_100g")/100.0,
    }

def classify_by_macros(row) -> str:
    m = per_gram(row)
    p,c,f = m["p"], m["c"], m["f"]
    if p >= c and p >= f and p >= 0.08: return "protein"
    if c >= p and c >= f and c >= 0.15: return "carb"
    if f >  c and f >  p and f  >= 0.08: return "fat"
    return "mixed"

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
    mode: str = "carbs"  # ou "kcal"

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

# ---------- Synonymes simples ----------
LIKE_MAP = {
    "chips": ["Croustilles", "Croustilles de maïs", "Chips"],
    "yogourt grec": ["Yogourt grec, nature, 0%", "Yogourt grec, nature, 2%"],
    "yogourt": ["Yogourt grec, nature, 0%", "Yogourt, nature"],
    "fruit": ["Pomme", "Banane", "Fraises", "Bleuets", "Raisin", "Orange"],
    "pâtes": ["Spaghetti, cuit", "Macaroni, cuit", "Pâtes, cuites"],
    "riz": ["Riz brun, grains longs, cuit", "Riz blanc à grains longs, cuit", "Riz sauvage, cuit"],
    "boeuf": ["Boeuf haché, extra maigre, émiétté, sauté", "Boeuf haché, extra maigre, cru"],
    "patate": ["Pomme de terre au four, chair et pelure", "Pomme de terre, bouillie, chair et pelure"],
    "haricot": ["Haricots noirs, en conserve, égouttés", "Haricots rouges, bouillis"],
    "jus de pomme": ["Jus de pomme, non sucré"],
}

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
            if contains: picked = contains[0]; break
            close = difflib.get_close_matches(k, low, n=1, cutoff=0.6)
            if close: picked = foods[low.index(close[0])]; break
        if picked and picked not in results:
            results.append(picked)
    return results

# ---------- Bornes plausibles (min = 0 pour éviter négatifs) ----------
def bounds_for(nm: str) -> Tuple[float, Optional[float]]:
    l = (nm or "").lower()
    if any(k in l for k in ["yogourt", "boeuf", "poulet", "dinde", "poisson", "tofu", "oeuf", "fromage"]):
        return (0.0, 300.0)     # protéine
    if any(k in l for k in ["pomme de terre", "spaghetti", "pâtes", "riz", "pain", "avoine", "quinoa"]):
        return (0.0, 400.0)     # glucides
    if any(k in l for k in ["croustille", "chips", "huile", "beurre", "amande", "noisette", "arachide", "chocolat"]):
        return (0.0, 60.0)      # snacks/gras denses
    return (0.0, 300.0)         # par défaut

# ---------- Suggestion (priorité prot -> carb -> fat) ----------
class SuggestReq(BaseModel):
    current: List[Dict[str, Any]]   # aliments à remplacer (libellé + g)
    likes:   List[str]              # envies (libellés libres)
    round_to: int = 5

@app.post("/suggest-plan")
def suggest_plan(req: SuggestReq):
    lazy_load()
    if LOAD_ERR:
        raise HTTPException(status_code=500, detail=f"DB load failed: {LOAD_ERR}")

    # 1) Cibles à atteindre (basé sur "current")
    if not req.current:
        raise HTTPException(status_code=422, detail={"error": "Aucun aliment fourni dans 'current'."})
    targets = totals_for_plan(req.current)  # on matche kcal ±10, prot ±3

    # 2) Candidats: uniquement ce que le client a écrit (avec synonymes)
    if not req.likes:
        raise HTTPException(status_code=422, detail={"error": "Aucune 'Envie' fournie."})
    cands = map_likes_to_exact_candidates(req.likes, FOODS)
    if not cands:
        raise HTTPException(status_code=422, detail={"error": "Aucun des 'Envies' n'a été reconnu."})

    # 3) Choisir au plus 1 aliment par catégorie, en respectant la priorité
    prot = carb = fat = None
    for nm in cands:
        r = row_for(nm)
        if r is None: continue
        cat = classify_by_macros(r)
        if cat == "protein" and prot is None: prot = nm
        elif cat == "carb" and carb is None: carb = nm
        elif cat == "fat" and fat is None: fat = nm
    # fallback: si pas de carb/fat, on accepte "mixed" au besoin
    if carb is None:
        for nm in cands:
            r = row_for(nm); 
            if r is None: continue
            if classify_by_macros(r) == "mixed": carb = nm; break
    if fat is None:
        for nm in cands:
            r = row_for(nm); 
            if r is None: continue
            if classify_by_macros(r) == "mixed": fat = nm; break

    if prot is None:
        raise HTTPException(status_code=422, detail={"error": "Aucune source de protéines reconnue dans 'Envies'."})

    # 4) Récup per-gram + bornes
    items = []
    for nm in [prot, carb, fat]:
        if nm is None: continue
        r = row_for(nm)
        g = per_gram(r)
        lo, hi = bounds_for(nm)
        items.append({"name": nm, "per": g, "lo": lo, "hi": hi, "g": 0.0})

    # 5) Étape A — régler la protéine d'abord
    targetP = targets["prot_g"]
    prot_item = items[0]  # toujours la protéine en premier
    pP, pK = prot_item["per"]["p"], prot_item["per"]["k"]
    if pP <= 0:
        raise HTTPException(status_code=422, detail={"error": f"La protéine '{prot_item['name']}' n'apporte pas de protéines exploitables."})
    gP = targetP / max(pP, 1e-9)
    gP = min(max(gP, prot_item["lo"]), prot_item["hi"])
    prot_item["g"] = gP

    # 6) Étape B — ajuster les kcal avec les glucides ensuite
    targetK = targets["kcal"]
    curK = pK * prot_item["g"]
    curP = pP * prot_item["g"]

    carb_item = next((x for x in items if x["name"] != prot_item["name"] and x["per"]["c"] > 0), None)
    if carb_item is not None:
        cK, cP = carb_item["per"]["k"], carb_item["per"]["p"]
        # grammes nécessaires par l'énergie restante
        needK = targetK - curK
        gC_energy = needK / max(cK, 1e-9)
        # fenêtre protéique restante (±3 g)
        minP = targetP - 3.0 - curP
        maxP = targetP + 3.0 - curP
        gC_min = 0.0 if cP <= 0 else max(0.0, minP / max(cP, 1e-9))
        gC_max = carb_item["hi"] if cP <= 0 else min(carb_item["hi"], maxP / max(cP, 1e-9))
        gC = min(max(gC_energy, max(carb_item["lo"], gC_min)), max(0.0, gC_max))
        carb_item["g"] = gC
        curK += cK * gC
        curP += cP * gC

    # 7) Étape C — finir au besoin avec les lipides
    fat_item = next((x for x in items if x["name"] != prot_item["name"] and (carb_item is None or x["name"] != carb_item["name"]) and x["per"]["f"] > 0), None)
    if fat_item is not None:
        fK, fP = fat_item["per"]["k"], fat_item["per"]["p"]
        needK = targetK - curK
        gF_energy = needK / max(fK, 1e-9)
        # contrainte prot ±3 g
        minP = targetP - 3.0 - curP
        maxP = targetP + 3.0 - curP
        gF_min = 0.0 if fP <= 0 else max(0.0, minP / max(fP, 1e-9))
        gF_max = fat_item["hi"] if fP <= 0 else min(fat_item["hi"], maxP / max(fP, 1e-9))
        gF = min(max(gF_energy, max(fat_item["lo"], gF_min)), max(0.0, gF_max))
        fat_item["g"] = gF
        curK += fK * gF
        curP += fP * gF

    # 8) Raffinement fin (±10 kcal / ±3 g prot) au pas de 1 g, sans jamais passer sous 0 ni au-dessus des bornes
    def score(kdiff, pdiff):
        # priorité: protéine puis kcal
        return 2.0*abs(pdiff/3.0) + 1.0*abs(kdiff/10.0)

    for _ in range(200):
        kdiff = targetK - curK
        pdiff = targetP - curP
        if abs(kdiff) <= 10 and abs(pdiff) <= 3:
            break

        # ordre d'ajustement: prot -> carb -> fat
        improved = False
        for item in items:
            per = item["per"]; lo, hi = item["lo"], item["hi"]
            # calcule l'effet d'un +1g ou -1g
            moves = []
            if item["g"] + 1 <= hi: moves.append(+1)
            if item["g"] - 1 >= lo: moves.append(-1)
            if not moves: continue
            best_move, best_score = 0, 1e9
            for mv in moves:
                newK = curK + per["k"]*mv
                newP = curP + per["p"]*mv
                sc = score(targetK - newK, targetP - newP)
                if sc < best_score:
                    best_score, best_move = sc, mv
            # n'applique que si ça améliore
            if best_score + 1e-9 < score(kdiff, pdiff):
                item["g"] += best_move
                curK += per["k"]*best_move
                curP += per["p"]*best_move
                improved = True
        if not improved:
            break

    # 9) Arrondi d'affichage (pas utilisateur), clamp final
    step = max(1, int(req.round_to))
    for it in items:
        it["g"] = float(max(it["lo"], min(it["hi"], round(it["g"]/step)*step)))

    suggestion = [{"aliment": it["name"], "grams": it["g"]} for it in items]
    final = totals_for_plan(suggestion)
    diff = {
        "kcal":   round(targets["kcal"]   - final["kcal"],   2),
        "prot_g": round(targets["prot_g"] - final["prot_g"], 2),
        "carb_g": round(targets["carb_g"] - final["carb_g"], 2),
        "fat_g":  round(targets["fat_g"]  - final["fat_g"],  2),
    }

    return {
        "scope": "replace_only_these_items",
        "current_totals": targets,
        "likes": req.likes,
        "chosen": [it["name"] for it in items],     # prot, carb, fat (si présents)
        "suggested_recipe": suggestion,             # grammes >= 0 et bornés
        "final_totals": final,
        "residual_diff": diff
    }

# ---------- Upload image (placeholder) ----------
@app.post("/upload-plan")
def upload_plan(file: UploadFile = File(...)):
    return {"filename": file.filename, "note": "OCR non implémenté encore."}
