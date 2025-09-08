# ====== Helpers (si pas déjà présents ou pour remplacer) ======
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
            v = float(row[col])
            return 0.0 if v != v else v
        except Exception:
            return 0.0
    return {
        "p": safe("protein_g_per_100g"),
        "c": safe("carb_g_per_100g"),
        "f": safe("fat_g_per_100g"),
        "k": safe("kcal_per_100g"),
        "fi": safe("fiber_g_per_100g"),
    }

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

def find_in_foods(keys: list[str], foods: list[str], limit=6) -> list[str]:
    out = []
    low = [f.lower() for f in foods]
    for key in keys:
        k = key.lower().strip()
        # contains
        hits = [foods[i] for i, nm in enumerate(low) if k in nm]
        for h in hits:
            if h not in out:
                out.append(h)
        # fuzzy
        import difflib
        close = difflib.get_close_matches(k, low, n=4, cutoff=0.6)
        for c in close:
            cand = foods[low.index(c)]
            if cand not in out:
                out.append(cand)
        if len(out) >= limit:
            break
    return out[:limit]

def choose_bucketed_candidates_from_likes(likes: list[str], foods: list[str], want=3) -> list[str]:
    # 1) collecte brute
    raw = []
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
    chosen = []
    if buckets["protein"]:
        chosen.append(buckets["protein"][0])
    else:
        for fb in ["Yogourt grec, nature, 0%", "Poulet, poitrine, sans peau, rôti", "Dinde hachée, extra maigre, émiéttée, sautée"]:
            bm = best_match(fb)
            if bm:
                chosen.append(bm); break

    # 4) compléter avec carb puis mixed/fat/protein (diversité)
    for cat in ["carb", "mixed", "fat", "protein"]:
        for nm in buckets[cat]:
            if nm not in chosen:
                chosen.append(nm)
            if len(chosen) >= want:
                break
        if len(chosen) >= want:
            break

    return chosen[:want]

def bounds_for(nm: str) -> tuple[float, float | None]:
    r = row_for(nm)
    cat = classify_by_macros(r) if r is not None else "mixed"
    if cat == "protein": return (90.0, 300.0)   # adaptables
    if cat == "carb":    return (90.0, 250.0)
    if cat == "fat":     return (10.0, 60.0)
    return (50.0, 250.0)  # mixed

# ====== Modèle d’entrée ======
class SuggestReq(BaseModel):
    current: List[Dict[str, Any]]   # <-- uniquement les aliments à remplacer (ce que le client a entré)
    likes:   List[str]              # ce qu’il veut manger à la place
    round_to: int = 5               # pas d’arrondi des grammes


# ====== Route principale (remplace ta /suggest-plan) ======
@app.post("/suggest-plan")
def suggest_plan(req: SuggestReq):
    lazy_load()
    if LOAD_ERR:
        raise HTTPException(status_code=500, detail=f"DB load failed: {LOAD_ERR}")

    try:
        # 1) Totaux de la PARTIE à remplacer (les items fournis)
        if not req.current:
            raise HTTPException(status_code=422, detail={"error": "Aucun aliment fourni dans 'current'."})
        cur_tot = totals_for_plan(req.current)  # cibles à égaler (±10 kcal / ±3 g prot)

        # 2) Candidats à partir des envies (avec garantie ≥1 protéine)
        cand = choose_bucketed_candidates_from_likes(req.likes, FOODS, want=3)

        names, P, K, B = [], [], [], []
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
            sol, *_ = np.linalg.lstsq(A2, b2, rcond=None)   # <<< IMPORTANT: pas de .T
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
            # fige les violés et soustrait leur contribution
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
        diff = {
            "kcal": round(cur_tot["kcal"] - final["kcal"], 2),
            "prot_g": round(cur_tot["prot_g"] - final["prot_g"], 2)
        }

        # micro-ajustements si hors tolérances
        for _ in range(30):
            okK = abs(diff["kcal"]) <= 10
            okP = abs(diff["prot_g"]) <= 3
            if okK and okP:
                break
            # cible la grandeur dominante
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
            diff = {
                "kcal": round(cur_tot["kcal"] - final["kcal"], 2),
                "prot_g": round(cur_tot["prot_g"] - final["prot_g"], 2)
            }

        return {
            "scope": "replace_only_these_items",
            "current_totals": cur_tot,           # cibles de la partie fournie
            "likes": req.likes,
            "candidates": names,
            "suggested_recipe": suggestion,      # ce qu’il doit manger à la place (avec grammes)
            "final_totals": final,               # totaux de la proposition
            "residual_diff": diff                # écart (doit tendre vers 0 et rester dans les tolérances)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail={"error": f"suggest-plan failed: {str(e)}"})
