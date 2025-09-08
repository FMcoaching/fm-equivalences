from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import difflib
from typing import Optional
from fm_equivalences import load_food_db, equivalent_portion

CSV = "nutrient_values_clean.csv"
db = load_food_db(CSV)
FOODS = sorted(db["Aliment"].dropna().unique().tolist())

app = FastAPI()

# sert /static/index.html
app.mount("/static", StaticFiles(directory="static"), name="static")

def best_match(name: str) -> Optional[str]:
    if not name:
        return None
    contains = [x for x in FOODS if name.lower() in x.lower()]
    if contains:
        return contains[0]
    close = difflib.get_close_matches(name, FOODS, n=1, cutoff=0.5)
    return close[0] if close else None

@app.get("/")
def root():
    # redirige la racine vers l'interface
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
        res["target_grams"] = round(res["target_grams"] / 5) * 5  # arrondi pratique
        return res
    except Exception as e:
        return {"error": str(e)}
