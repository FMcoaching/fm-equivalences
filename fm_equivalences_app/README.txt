
# FM Équivalences — Mini‑app (Guide pas-à-pas)

Ce dossier contient tout le nécessaire pour lancer une mini‑application qui calcule des équivalences alimentaires
(iso‑glucides ou iso‑kcal) à partir d'une base d'aliments (CSV).
L'upload d'une photo de plan (OCR) est **optionnel**.

## Contenu
- app.py — serveur FastAPI (API + page web statique)
- fm_equivalences.py — logique d'équivalence
- nutrient_values_clean.csv — base d'aliments (déjà nettoyée)
- static/index.html — interface web simple
- requirements.txt — dépendances Python

---

## Étapes (Windows 10/11)

1) **Installer Python 3.10+**
   - Si vous n'avez pas Python : installez‑le depuis le site officiel (cochez “Add Python to PATH”).

2) **Télécharger le dossier**
   - Dézippez l'archive `fm_equivalences_app.zip` (ou copiez ce dossier) dans, par exemple, `C:\Users\<votre_nom>\FMapp`.

3) **Ouvrir PowerShell dans le dossier**
   - Clic droit dans le dossier → “Ouvrir dans le Terminal”
   - Ou: Menu Démarrer → tapez “PowerShell”, puis `cd` vers le dossier.
     Exemple :
       cd "C:\Users\<votre_nom>\FMapp\fm_equivalences_app"

4) **Créer un environnement virtuel (recommandé)**
   python -m venv .venv
   .\.venv\Scripts\activate

5) **Installer les dépendances**
   pip install -r requirements.txt

6) **(Option OCR)** Installer Tesseract (si vous voulez l'upload de photo)
   - Recherchez “Tesseract OCR Windows” et installez‑le.
   - Après installation, redémarrez le Terminal si nécessaire.

7) **Lancer le serveur**
   uvicorn app:app --reload --port 8000

8) **Ouvrir l’interface**
   - Dans votre navigateur : http://localhost:8000/static/index.html

---

## Étapes (macOS)

1) **Python 3.10+** (via Python.org ou Homebrew)
2) **Dézipper ce dossier** (par ex. dans `~/FMapp/fm_equivalences_app`)
3) **Terminal → aller dans le dossier**
   cd ~/FMapp/fm_equivalences_app
4) **Environnement virtuel (recommandé)**
   python3 -m venv .venv
   source .venv/bin/activate
5) **Installer les dépendances**
   pip install -r requirements.txt
6) **(Option OCR)** Installer Tesseract (via Homebrew : `brew install tesseract`)
7) **Lancer le serveur**
   uvicorn app:app --reload --port 8000
8) **Ouvrir l’interface**
   http://localhost:8000/static/index.html

---

## Utilisation rapide

- **Calcul direct** :
  - Tapez un nom d'aliment “Source” (ex: “Riz blanc à grains longs, cuit”), entrez une quantité en grammes (ex: 100 g),
    puis un “Cible” (ex: “Spaghetti, cuit”), choix du mode :
    - “Iso‑glucides” → égalise les glucides
    - “Iso‑kcal” → égalise les calories

- **Photo du plan (OCR)** :
  - Cliquez “Analyser” et chargez une photo; l’app tentera d’identifier *aliment + quantité*.
  - Si Tesseract n’est pas installé, un message l’indiquera; le calcul direct reste disponible.

---

## Déploiement en ligne (optionnel, plus tard)
- Render.com ou Railway.app : créez un nouveau service “Web” Python.
- Commande de démarrage :
  uvicorn app:app --host 0.0.0.0 --port $PORT
- Ajoutez les fichiers du dossier au dépôt (Git) et déployez.

Bon démarrage !
