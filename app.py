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
        return (0.0, 300.0)     # protéine
    if any(k in l for k in ["pomme de terre", "spaghetti", "pâtes", "riz", "pain", "avoine", "quinoa"]):
        return (0.0, 400.0)     # glucides
    if any(k in l for k in ["croustille", "chips", "huile", "beurre", "amande", "noisette", "arachide", "chocolat"]):
        return (0.0, 60.0)      # snacks/gras denses
    return (0.0, 300.0)         # par défaut

# ===================== /mix-plan =====================

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

def _mk_rows(names: List[str]):
    rows = []
    for nm in names[:3]:
        r = row_for(nm)
        if r is None:
            continue
        per = per_gram(r)
        lo, hi = bounds_for(nm)
        rows.append({"name": nm, "per": per, "lo": lo, "hi": hi, "g": 0.0})
    return rows

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
      4) Lipides : si zéro budget lipides, réduire automatiquement des glucides pour libérer des kcal,
         sinon message d’impossibilité
    """
    lazy_load()
    if LOAD_ERR:
        raise HTTPException(status_code=500, detail=f"DB load failed: {LOAD_ERR}")

    T = req.targets
    if T.prot_g <= 0 or T.carb_g <= 0 or T.fat_g <= 0:
        raise HTTPException(status_code=422, detail={"error": "Les trois macros (prot, gluc, lip) doivent être > 0 g."})

    prot_rows = _mk_rows(req.foods.protein)
    carb_rows = _mk_rows(req.foods.carb)
    fat_rows  = _mk_rows(req.foods.fat)

    if not prot_rows or all(r["per"]["p"] <= 1e-12 for r in prot_rows):
        raise HTTPException(status_code=422, detail={"error": "Aucune vraie source de protéines (ligne Protéines)."})
    if not carb_rows or all(r["per"]["c"] <= 1e-12 for r in carb_rows):
        raise HTTPException(status_code=422, detail={"error": "Aucune vraie source de glucides (ligne Glucides)."})
    if not fat_rows  or all(r["per"]["f"] <= 1e-12 for r in fat_rows):
        raise HTTPException(status_code=422, detail={"error": "Aucune vraie source de lipides (ligne Lipides)."})

    # ---- 1) PROTÉINES ----
    prot_rows.sort(key=lambda r: r["per"]["p"], reverse=True)
    prot_target = T.prot_g
    for r in prot_rows:
        if prot_target <= 1e-9:
            break
        ppg = r["per"]["p"]
        if ppg <= 1e-12:
            continue
        need_g = prot_target / ppg
        g = _clamp(need_g, r["lo"], r["hi"])
        r["g"] = g
        prot_target -= ppg * g
    if prot_target > 0.75:
        raise HTTPException(status_code=422, detail={"error": "Impossible d’atteindre la protéine cible avec ces aliments et bornes."})

    curK = sum(r["per"]["k"]*r["g"] for r in prot_rows)
    curP = sum(r["per"]["p"]*r["g"] for r in prot_rows)
    curC = sum(r["per"]["c"]*r["g"] for r in prot_rows)
    curF = sum(r["per"]["f"]*r["g"] for r in prot_rows)

    # ---- 2) GLUCIDES ----
    carb_rows.sort(key=lambda r: (r["per"]["c"], -r["per"]["p"]), reverse=True)
    carb_target = max(0.0, T.carb_g - curC)
    for r in carb_rows:
        if carb_target <= 1e-9:
            break
        cpg = r["per"]["c"]
        if cpg <= 1e-12:
            continue
        need_g = carb_target / cpg
        g = _clamp(need_g, r["lo"], r["hi"])
        r["g"] = g
        carb_target -= cpg * g
    if carb_target > 0.75:
        raise HTTPException(status_code=422, detail={"error": "Impossible d’atteindre les glucides cibles avec ces aliments (ou bornes trop strictes)."})

    curK += sum(r["per"]["k"]*r["g"] for r in carb_rows)
    curP += sum(r["per"]["p"]*r["g"] for r in carb_rows)
    curC += sum(r["per"]["c"]*r["g"] for r in carb_rows)
    curF += sum(r["per"]["f"]*r["g"] for r in carb_rows)

    # ---- 3) RÉAJUSTER PROT si les glucides ont ajouté de la prot ----
    overP = curP - T.prot_g
    if overP > 0.5:
        # Réduire d'abord la protéine la moins efficiente (kcal/gramme prot le + élevé)
        def kcal_per_prot(r):
            return (r["per"]["k"] / max(r["per"]["p"], 1e-9)) if r["per"]["p"] > 0 else 1e9
        prot_rows.sort(key=kcal_per_prot, reverse=True)
        for r in prot_rows:
            if overP <= 1e-9:
                break
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

    # ---- 4) LIPIDES (avec auto-réduction des glucides si nécessaire) ----
    fat_rows.sort(key=lambda r: (r["per"]["f"], -r["per"]["p"]), reverse=True)
    fat_target = max(0.0, T.fat_g - curF)

    if fat_target <= 1e-6:
        # Zéro budget lipides —> libérer des kcal en réduisant des glucides
        need_fat_g = max(0.0, T.fat_g - curF)
        if need_fat_g > 0:
            kcal_needed = need_fat_g * 9.0
            carb_macro_to_reduce = kcal_needed / 4.0  # 1 g gluc = 4 kcal

            # Réduire d'abord les glucides les plus "purs": peu de prot, beaucoup de carb
            carb_rows.sort(key=lambda r: (r["per"]["p"], -r["per"]["c"]))
            remaining = carb_macro_to_reduce
            for r in carb_rows:
                if remaining <= 1e-6:
                    break
                cpg = r["per"]["c"]
                if cpg <= 1e-12 or r["g"] <= r["lo"] + 1e-9:
                    continue
                max_c_macro_removable = cpg * (r["g"] - r["lo"])
                take_c_macro = min(remaining, max_c_macro_removable)
                if take_c_macro <= 0:
                    continue
                dg = take_c_macro / cpg
                r["g"] -= dg
                # maj totaux
                curC -= take_c_macro
                curK -= r["per"]["k"] * dg
                curP -= r["per"]["p"] * dg
                curF -= r["per"]["f"] * dg
                remaining -= take_c_macro

            if remaining > 1e-3:
                # Impossible de libérer assez de kcal via glucides
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "Aucun budget de lipides et réduction de glucides insuffisante (bornes).",
                        "hint": "Choisis des glucides plus 'purs' (riz/pommes de terre) ou augmente les bornes."
                    }
                )

            # Ajouter maintenant les lipides (kcal constantes)
            fat_to_add = need_fat_g
            for r in fat_rows:
                if fat_to_add <= 1e-6:
                    break
                fpg = r["per"]["f"]
                if fpg <= 1e-12:
                    continue
                need_g = fat_to_add / fpg
                headroom = (r["hi"] - r["g"]) if r["hi"] is not None else 1e9
                dg = min(need_g, max(0.0, headroom))
                if dg <= 0:
                    continue
                r["g"] += dg
                curF += fpg * dg
                curK += r["per"]["k"] * dg
                fat_to_add -= fpg * dg

            if fat_to_add > 1e-3:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "Impossible d’ajouter assez de lipides après réduction des glucides (bornes).",
                        "hint": "Choisis une source de lipides plus dense (huile/noix) ou augmente les bornes."
                    }
                )

            # Si la réduction des glucides a trop entamé la protéine, remonte-la et recompense en glucides
            prot_short = T.prot_g - curP  # si positif, il manque de prot
            if prot_short > 0.5:
                def eff(r):  # priorité aux aliments prot les plus efficients (kcal/prot faible)
                    return (r["per"]["k"] / max(r["per"]["p"], 1e-9)) if r["per"]["p"] > 0 else 1e9
                prot_rows.sort(key=eff)
                remaining_p = prot_short
                for r in prot_rows:
                    if remaining_p <= 1e-6:
                        break
                    ppg = r["per"]["p"]
                    if ppg <= 1e-12:
                        continue
                    headroom = (r["hi"] - r["g"]) if r["hi"] is not None else 1e9
                    if headroom <= 1e-9:
                        continue
                    dg = min(remaining_p / ppg, headroom)
                    r["g"] += dg
                    curP += ppg * dg
                    curK += r["per"]["k"] * dg
                    curC += r["per"]["c"] * dg
                    curF += r["per"]["f"] * dg
                    remaining_p -= ppg * dg

                # Pour garder les kcal constantes: si surplus de kcal, re-réduire un peu de glucides
                kcal_expected = 4*T.prot_g + 4*T.carb_g + 9*T.fat_g
                kcal_over = curK - kcal_expected
                if kcal_over > 0.5:
                    c_macro_to_remove = kcal_over / 4.0
                    carb_rows.sort(key=lambda r: (r["per"]["p"], -r["per"]["c"]))
                    rem = c_macro_to_remove
                    for r in carb_rows:
                        if rem <= 1e-6:
                            break
                        cpg = r["per"]["c"]
                        if cpg <= 1e-12 or r["g"] <= r["lo"] + 1e-9:
                            continue
                        max_c_macro_removable = cpg * (r["g"] - r["lo"])
                        take = min(rem, max_c_macro_removable)
                        if take <= 0:
                            continue
                        dg = take / cpg
                        r["g"] -= dg
                        curC -= take
                        curK -= r["per"]["k"] * dg
                        curP -= r["per"]["p"] * dg
                        curF -= r["per"]["f"] * dg
                        rem -= take
                    if rem > 1e-3:
                        raise HTTPException(
                            status_code=422,
                            detail={
                                "error": "Ajustement automatique impossible sans violer les bornes.",
                                "hint": "Choisis des glucides plus 'purs' ou élargis les bornes."
                            }
                        )

    else:
        # Cas normal : on a du budget lipides
        for r in fat_rows:
            if fat_target <= 1e-9:
                break
            fpg = r["per"]["f"]
            if fpg <= 1e-12:
                continue
            need_g = fat_target / fpg
            headroom = (r["hi"] - r["g"]) if r["hi"] is not None else 1e9
            dg = min(need_g, max(0.0, headroom))
            if dg <= 0:
                continue
            r["g"] += dg
            curF += fpg * dg
            curK += r["per"]["k"] * dg
            fat_target -= fpg * dg

        if fat_target > 1e-3:
            raise HTTPException(status_code=422, detail={
                "error": "Impossible d’atteindre les lipides cibles (bornes).",
                "hint": "Choisir une source de lipides plus dense (huile/noix) ou augmenter les bornes."
            })

    # ---- Assemblage & sortie ----
    step = max(1, int(req.round_to))
    suggestion = []
    for rows in (prot_rows, carb_rows, fat_rows):
        for r in rows:
            g = float(_clamp(round(r["g"]/step)*step, r["lo"], r["hi"] if r["hi"] is not None else 1e9))
            if g > 0:
                suggestion.append({"aliment": r["name"], "grams": g})

    final = totals_for_plan(suggestion)
    kcal_expected = round(4*T.prot_g + 4*T.carb_g + 9*T.fat_g, 2)

    return {
        "targets": {"prot_g": T.prot_g, "carb_g": T.carb_g, "fat_g": T.fat_g, "kcal_expected": kcal_expected},
        "suggested_recipe": suggestion,
        "final_totals": final,
        "note": "Ordre: prot → glucides → réajust prot → lipides. Si lipides initialement impossibles, réduction automatique des glucides pour libérer des kcal."
    }

# ---------- Upload image (placeholder) ----------
@app.post("/upload-plan")
def upload_plan(file: UploadFile = File(...)):
    return {"filename": file.filename, "note": "OCR non implémenté encore."}
