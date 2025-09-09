# app.py — FM Équivalences (stable)
# - Equivalence simple (/equivalence)
# - Mix plan par macros et aliments par macro (/mix-plan)
# Ordre logique: Prot -> Glucides -> Réajust Prot -> Lipides
# Si zéro budget lipides: réduction automatique des glucides pour libérer des kcal, sinon message.

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
import difflib
import numpy as np

# --------- Lazy load DB + utilitaires ---------
db = None
FOODS: List[str] = []
LOAD_ERR: Optional[str] = None

def lazy_load():
    """Charge la base CSV une fois."""
    global db, FOODS, LOAD_ERR, equivalent_portion, load_food_db
    if db is not None or LOAD_ERR is not None:
        return
    try:
        from fm_equivalences import load_food_db, equivalent_portion
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
    """Retourne la ligne (pandas Series) pour le meilleur libellé."""
    lazy_load()
    if LOAD_ERR or db is None:
        return None
    m = best_match(name)
    if not m:
        return None
    rows = db[db["Aliment"] == m]
    return rows.iloc[0] if not rows.empty else None

def _safe_float(row, col: str) -> float:
    try:
        v = float(row[col])
        return v if v == v else 0.0
    except Exception:
        return 0.0

def macros_for_grams(row, grams: float) -> Dict[str, float]:
    f = grams / 100.0
    return {
        "kcal":   round(_safe_float(row, "kcal_per_100g")      * f, 4),
        "prot_g": round(_safe_float(row, "protein_g_per_100g") * f, 4),
        "carb_g": round(_safe_float(row, "carb_g_per_100g")    * f, 4),
        "fat_g":  round(_safe_float(row, "fat_g_per_100g")     * f, 4),
        "fiber_g":round(_safe_float(row, "fiber_g_per_100g")   * f, 4),
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

def per_gram(row) -> Dict[str, float]:
    return {
        "p": _safe_float(row, "protein_g_per_100g")/100.0,
        "c": _safe_float(row, "carb_g_per_100g")/100.0,
        "f": _safe_float(row, "fat_g_per_100g")/100.0,
        "k": _safe_float(row, "kcal_per_100g")/100.0,
    }

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
        res["target_grams"] = round(res["target_grams"] / 5) * 5
        return res
    except Exception as e:
        raise HTTPException(status_code=422, detail={"error": str(e)})

# --------- Bornes plausibles (évite négatifs & quantités aberrantes) ---------
def bounds_for(nm: str) -> Tuple[float, Optional[float]]:
    l = (nm or "").lower()
    if any(k in l for k in ["yogourt", "boeuf", "poulet", "dinde", "poisson", "tofu", "oeuf", "fromage"]):
        return (0.0, 400.0)     # protéine
    if any(k in l for k in ["pomme de terre", "spaghetti", "pâtes", "riz", "pain", "avoine", "quinoa"]):
        return (0.0, 400.0)     # glucides
    if any(k in l for k in ["croustille", "chips", "huile", "beurre", "amande", "noisette", "arachide", "chocolat"]):
        return (0.0, 60.0)      # snacks/gras denses
    return (0.0, 300.0)         # par défaut

# ===================== /mix-plan (2 aliments possibles) =====================

class MixTargets(BaseModel):
    prot_g: float = Field(..., ge=0)
    carb_g: float = Field(..., ge=0)
    fat_g:  float = Field(..., ge=0)

class MixFoods(BaseModel):
    protein: List[str] = []   # 0..3
    carb:    List[str] = []   # 0..3
    fat:     List[str] = []   # 0..3

class MixReq(BaseModel):
    targets: MixTargets
    foods:   MixFoods
    round_to: int = 5

def _mk_all_rows(foods: MixFoods):
    """Construit un dict {name -> rowObj unique} pour TOUS les aliments fournis,
    peu importe le groupe (prot/carb/fat). Chaque rowObj contient grammes 'g' partagés."""
    names = []
    for lst in (foods.protein[:3], foods.carb[:3], foods.fat[:3]):
        for nm in lst:
            nm = nm.strip()
            if nm and nm not in names:
                names.append(nm)

    rows_by_name = {}
    for nm in names:
        r = row_for(nm)
        if r is None:
            continue
        per = per_gram(r)
        lo, hi = bounds_for(nm)
        rows_by_name[nm] = {"name": nm, "per": per, "lo": lo, "hi": hi, "g": 0.0}
    return rows_by_name

def _capability(rows, macro_key: str) -> float:
    """Capacité maximale théorique du macro sur l’ensemble des aliments (en g de macro)."""
    cap = 0.0
    for r in rows.values():
        per = r["per"][macro_key]
        if per <= 1e-12: 
            continue
        hi = r["hi"] if r["hi"] is not None else 1e9
        cap += per * max(0.0, hi - r["g"])
    return cap

def _clamp(v, lo, hi):
    v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v

@app.post("/mix-plan")
def mix_plan(req: MixReq):
    """
    Ordre logique :
      1) Protéines (maximiser pour atteindre T.prot_g)
      2) Glucides (atteindre T.carb_g)
      3) Réajuster protéine si les glucides ont ajouté de la prot
      4) Lipides : si zéro budget lipides, réduire automatiquement des glucides pour libérer des kcal;
         sinon message d’impossibilité
      + Finition: micro-ajustements pour <= 0,5 g d'écart par macro si possible
    """
    lazy_load()
    if LOAD_ERR:
        raise HTTPException(status_code=500, detail=f"DB load failed: {LOAD_ERR}")

    T = req.targets
    if T.prot_g <= 0 or T.carb_g <= 0 or T.fat_g <= 0:
        raise HTTPException(status_code=422, detail={"error": "Les trois macros (prot, gluc, lip) doivent être > 0 g."})

    # Rows uniques (2 aliments au total autorisés)
    rows = _mk_all_rows(req.foods)
    if len(rows) < 2:
        raise HTTPException(status_code=422, detail={"error": "Ajoute au moins 2 aliments (au total)."})
    # Pointeurs de groupe (référencent les mêmes objets)
    def pick(names): return [rows[n] for n in names if n in rows]
    prot_rows = pick(req.foods.protein)
    carb_rows = pick(req.foods.carb)
    fat_rows  = pick(req.foods.fat)
    all_rows  = list(rows.values())

    # tri utilitaires
    def by(desc_key):
        return sorted(all_rows, key=lambda r: r["per"][desc_key], reverse=True)

    # --------- 1) PROTÉINES ---------
    # Si aucune "prot" fournie, on prend les meilleurs contributeurs de prot parmi tous les aliments
    P_cands = prot_rows if len(prot_rows)>0 else by("p")
    prot_target = T.prot_g
    for r in P_cands:
        if prot_target <= 1e-9: break
        ppg = r["per"]["p"]
        if ppg <= 1e-12: continue
        need_g = prot_target / ppg
        g = _clamp(need_g + r["g"], r["lo"], r["hi"])  # cumulatif si r déjà utilisé
        dg = g - r["g"]
        r["g"] = g
        prot_target -= ppg * dg
    if prot_target > 0.75:
        # Vérifie si c’est un vrai manque de capacité
        if _capability(rows, "p") < T.prot_g - (T.prot_g - prot_target):
            raise HTTPException(status_code=422, detail={"error": "Capacité protéique insuffisante avec ces aliments/bornes."})
        raise HTTPException(status_code=422, detail={"error": "Impossible d’atteindre la protéine cible avec ces aliments et bornes."})

    # Totaux courants
    curK = sum(r["per"]["k"]*r["g"] for r in all_rows)
    curP = sum(r["per"]["p"]*r["g"] for r in all_rows)
    curC = sum(r["per"]["c"]*r["g"] for r in all_rows)
    curF = sum(r["per"]["f"]*r["g"] for r in all_rows)

    # --------- 2) GLUCIDES ---------
    C_cands = carb_rows if len(carb_rows)>0 else by("c")
    carb_target = max(0.0, T.carb_g - curC)
    for r in C_cands:
        if carb_target <= 1e-9: break
        cpg = r["per"]["c"]
        if cpg <= 1e-12: continue
        need_g = carb_target / cpg
        g = _clamp(need_g + r["g"], r["lo"], r["hi"])
        dg = g - r["g"]
        r["g"] = g
        curC += cpg * dg
        curP += r["per"]["p"] * dg
        curK += r["per"]["k"] * dg
        curF += r["per"]["f"] * dg
        carb_target = max(0.0, T.carb_g - curC)
    if carb_target > 0.75:
        if _capability(rows, "c") < T.carb_g:
            raise HTTPException(status_code=422, detail={"error": "Capacité glucides insuffisante avec ces aliments/bornes."})
        raise HTTPException(status_code=422, detail={"error": "Impossible d’atteindre les glucides cibles avec ces aliments (bornes)."})
    
    # --------- 3) RÉAJUSTER PROT si les glucides ont ajouté de la prot ---------
    overP = curP - T.prot_g
    if overP > 0.5:
        # Réduire d’abord l’aliment avec le plus mauvais ratio (kcal/prot élevé)
        def kcal_per_prot(r): 
            return (r["per"]["k"] / max(r["per"]["p"], 1e-9)) if r["per"]["p"]>0 else 1e9
        P_red = sorted(P_cands, key=kcal_per_prot, reverse=True)
        for r in P_red:
            if overP <= 1e-9: break
            ppg = r["per"]["p"]
            if ppg <= 1e-12 or r["g"] <= r["lo"] + 1e-9: 
                continue
            reduc_g = min((overP / ppg), r["g"] - r["lo"])
            r["g"] -= reduc_g
            curP -= ppg * reduc_g
            curK -= r["per"]["k"] * reduc_g
            curC -= r["per"]["c"] * reduc_g
            curF -= r["per"]["f"] * reduc_g
            overP = curP - T.prot_g
        if overP > 0.75:
            raise HTTPException(status_code=422, detail={"error": "Impossible de réajuster la protéine (bornes). Choisis des glucides moins protéiques."})

    # --------- 4) LIPIDES (avec auto-réduction glucides si nécessaire) ---------
    F_cands = fat_rows if len(fat_rows)>0 else by("f")
    fat_target = max(0.0, T.fat_g - curF)

    if fat_target <= 1e-6:
        # Zéro budget lipides -> libérer des kcal via glucides
        need_fat_g = max(0.0, T.fat_g - curF)
        if need_fat_g > 0:
            kcal_needed = need_fat_g * 9.0
            carb_macro_to_reduce = kcal_needed / 4.0

            # Réduire d’abord les glucides les plus “purs” (peu de prot)
            C_red = sorted(C_cands, key=lambda r: (r["per"]["p"], -r["per"]["c"]))
            remaining = carb_macro_to_reduce
            for r in C_red:
                if remaining <= 1e-6: break
                cpg = r["per"]["c"]
                if cpg <= 1e-12 or r["g"] <= r["lo"] + 1e-9: 
                    continue
                max_c_macro_removable = cpg * (r["g"] - r["lo"])
                take_c_macro = min(remaining, max_c_macro_removable)
                if take_c_macro <= 0: 
                    continue
                dg = take_c_macro / cpg
                r["g"] -= dg
                curC -= take_c_macro
                curK -= r["per"]["k"] * dg
                curP -= r["per"]["p"] * dg
                curF -= r["per"]["f"] * dg
                remaining -= take_c_macro

            if remaining > 1e-3:
                raise HTTPException(status_code=422, detail={
                    "error": "Aucun budget lipides et réduction de glucides insuffisante (bornes).",
                    "hint": "Ajoute une vraie source de lipides (huile/noix) ou change tes glucides pour plus 'purs'."
                })

            # Ajouter maintenant les lipides
            fat_to_add = need_fat_g
            for r in F_cands:
                if fat_to_add <= 1e-6: break
                fpg = r["per"]["f"]
                if fpg <= 1e-12: continue
                need_g = fat_to_add / fpg
                headroom = (r["hi"] - r["g"]) if r["hi"] is not None else 1e9
                dg = min(need_g, max(0.0, headroom))
                if dg <= 0: continue
                r["g"] += dg
                curF += fpg * dg
                curK += r["per"]["k"] * dg
                fat_to_add -= fpg * dg

            if fat_to_add > 1e-3:
                raise HTTPException(status_code=422, detail={
                    "error": "Impossible d’ajouter assez de lipides après réduction des glucides (bornes).",
                    "hint": "Choisis une source de lipides plus dense (huile/noix) ou augmente les bornes."
                })

            # Si on a perdu trop de prot en réduisant les glucides → remonter via meilleurs ppg (parmi TOUS les aliments)
            prot_short = T.prot_g - curP
            if prot_short > 0.5:
                P_up = by("p")  # meilleurs contributeurs de prot possibles (tous)
                rem = prot_short
                for r in P_up:
                    if rem <= 1e-6: break
                    ppg = r["per"]["p"]
                    if ppg <= 1e-12: continue
                    headroom = (r["hi"] - r["g"]) if r["hi"] is not None else 1e9
                    if headroom <= 1e-9: continue
                    dg = min(rem / ppg, headroom)
                    r["g"] += dg
                    curP += ppg * dg
                    curK += r["per"]["k"] * dg
                    curC += r["per"]["c"] * dg
                    curF += r["per"]["f"] * dg
                    rem -= ppg * dg

                # Garder les kcal constantes : si surplus kcal, re-réduire un peu de glucides
                kcal_expected = 4*T.prot_g + 4*T.carb_g + 9*T.fat_g
                kcal_over = curK - kcal_expected
                if kcal_over > 0.5:
                    C_red2 = sorted(C_cands, key=lambda r: (r["per"]["p"], -r["per"]["c"]))
                    remC = kcal_over / 4.0
                    for r in C_red2:
                        if remC <= 1e-6: break
                        cpg = r["per"]["c"]
                        if cpg <= 1e-12 or r["g"] <= r["lo"] + 1e-9: 
                            continue
                        max_c_macro_removable = cpg * (r["g"] - r["lo"])
                        take = min(remC, max_c_macro_removable)
                        if take <= 0: continue
                        dg = take / cpg
                        r["g"] -= dg
                        curC -= take
                        curK -= r["per"]["k"] * dg
                        curP -= r["per"]["p"] * dg
                        curF -= r["per"]["f"] * dg
                        remC -= take

    else:
        # Cas normal : on a un budget lipides
        for r in F_cands:
            if fat_target <= 1e-9: break
            fpg = r["per"]["f"]
            if fpg <= 1e-12: continue
            need_g = fat_target / fpg
            headroom = (r["hi"] - r["g"]) if r["hi"] is not None else 1e9
            dg = min(need_g, max(0.0, headroom))
            if dg <= 0: continue
            r["g"] += dg
            curF += fpg * dg
            curK += r["per"]["k"] * dg
            fat_target -= fpg * dg
        if fat_target > 1e-3:
            raise HTTPException(status_code=422, detail={
                "error": "Impossible d’atteindre les lipides cibles (bornes).",
                "hint": "Ajoute une vraie source de lipides (huile/noix) ou augmente les bornes."
            })

    # --------- 5) Micro-finition (±0,5 g par macro si possible) ---------
    # Boucle courte : on corrige la macro la plus en écart, 1 g d’aliment à la fois, dans les bornes.
    for _ in range(400):
        dP = T.prot_g - curP
        dC = T.carb_g - curC
        dF = T.fat_g  - curF
        if abs(dP)<=0.5 and abs(dC)<=0.5 and abs(dF)<=0.5:
            break
        # choisir la dimension la plus en écart relatif
        key, diff = max([("p", abs(dP)/0.5), ("c", abs(dC)/0.5), ("f", abs(dF)/0.5)], key=lambda x:x[1])
        need_inc = ( (dP>0 and key=="p") or (dC>0 and key=="c") or (dF>0 and key=="f") )

        # candidat le plus efficace pour cette macro
        cand = None
        if need_inc:
            cand = max(all_rows, key=lambda r: r["per"][key])
        else:
            # réduire celui qui a le plus de macro par g (on enlève 1 g)
            cand = max(all_rows, key=lambda r: r["per"][key])

        if cand is None or cand["per"][key] <= 1e-12:
            break

        step = 1.0 if need_inc else -1.0
        ng = cand["g"] + step
        if ng < cand["lo"] or (cand["hi"] is not None and ng > cand["hi"]):
            # pas possible sur ce cand, essaie le suivant
            ordered = sorted(all_rows, key=lambda r: r["per"][key], reverse=True)
            moved = False
            for r in ordered:
                if need_inc and r["per"][key] <= 1e-12: 
                    continue
                ng2 = r["g"] + step
                if ng2 < r["lo"] or (r["hi"] is not None and ng2 > r["hi"]):
                    continue
                # applique
                r["g"] = ng2
                curP += r["per"]["p"] * step
                curC += r["per"]["c"] * step
                curF += r["per"]["f"] * step
                curK += r["per"]["k"] * step
                moved = True
                break
            if not moved:
                break
        else:
            cand["g"] = ng
            curP += cand["per"]["p"] * step
            curC += cand["per"]["c"] * step
            curF += cand["per"]["f"] * step
            curK += cand["per"]["k"] * step

    # ---- Assemblage & sortie ----
    step = max(1, int(req.round_to))
    suggestion = []
    for r in all_rows:
        g = float(_clamp(round(r["g"]/step)*step, r["lo"], r["hi"] if r["hi"] is not None else 1e9))
        if g > 0:
            suggestion.append({"aliment": r["name"], "grams": g})

    final = totals_for_plan(suggestion)
    kcal_expected = round(4*T.prot_g + 4*T.carb_g + 9*T.fat_g, 2)
    return {
        "targets": {"prot_g": T.prot_g, "carb_g": T.carb_g, "fat_g": T.fat_g, "kcal_expected": kcal_expected},
        "suggested_recipe": suggestion,
        "final_totals": final,
        "note": "Ordre: prot → glucides → réajust prot → lipides. Accepte 2 aliments total; réduit auto les glucides si lipides indisponibles."
    }

# ---------- Upload image (placeholder) ----------
@app.post("/upload-plan")
def upload_plan(file: UploadFile = File(...)):
    return {"filename": file.filename, "note": "OCR non implémenté encore."}
