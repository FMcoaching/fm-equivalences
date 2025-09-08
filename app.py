
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import RedirectResponse
import pandas as pd, difflib, io

# Dependencies for OCR (optional)
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

from fm_equivalences import load_food_db, equivalent_portion

CSV = "nutrient_values_clean.csv"
db = load_food_db(CSV)
FOODS = sorted(db["Aliment"].dropna().unique().tolist())

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

def best_match(name: str) -> Optional[str]:
    if not name: return None
    contains = [x for x in FOODS if name.lower() in x.lower()]
    if contains: return contains[0]
    close = difflib.get_close_matches(name, FOODS, n=1, cutoff=0.5)
    return close[0] if close else None

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

@app.post("/upload-plan")
def upload_plan(file: UploadFile = File(...)):
    if not OCR_AVAILABLE:
        return {"error": "OCR indisponible. Installez Tesseract + pillow + pytesseract pour activer cette fonction."}
    try:
        image_bytes = file.file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Utilise français; ajoutez '+eng' si plan mix FR/EN
        raw_text = pytesseract.image_to_string(image, lang="fra")
        # Lignes non vides
        lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
        # Extraction très simple: tentative d'identifier 'aliment' + 'quantité'
        import re
        parsed = []
        for l in lines:
            lower = l.lower().replace(",", ".")
            qty = None
            m = re.search(r"(\d+(\.\d+)?)\s*(g|grammes?)\b", lower) or re.search(r"(\d+(\.\d+)?)\s*(ml|mL)\b", lower)
            if m:
                qty = float(m.group(1))
            name = re.sub(r"(\d+(\.\d+)?\s*(g|grammes?|ml|mL))", "", lower).strip()
            name = name.replace(" de ", " ").replace(" d'", " ").strip()
            match = best_match(name)
            parsed.append({"raw": l, "aliment_detecte": name or None, "quantite": qty, "match_base": match})
        return {"items": parsed, "ocr_text": raw_text}
    except Exception as e:
        return {"error": str(e)}
        @app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")

