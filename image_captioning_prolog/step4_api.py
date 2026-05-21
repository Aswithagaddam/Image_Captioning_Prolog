# ============================================================
# STEP 4 — FastAPI Backend with Full Prolog Integration
# ============================================================
# HOW TO RUN:
#   pip install fastapi uvicorn python-multipart transformers torch Pillow spacy pyswip
#   python -m spacy download en_core_web_sm
#   apt-get install swi-prolog          (Linux)
#   brew install swi-prolog             (macOS)
#   uvicorn step4_api:app --reload --host 0.0.0.0 --port 8000
#
# API endpoints:
#   POST /caption          <- upload image file, get caption + Prolog output
#   POST /caption-url      <- pass image URL instead of file
#   GET  /prolog/query     <- run a custom Prolog query on a previously processed image
#   GET  /health
#   GET  /docs             <- Swagger UI
# ============================================================

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import spacy
import re
import io
import time
import os
import uuid

# ============================================================
# APP SETUP
# ============================================================

app = FastAPI(
    title="AI Image Captioning API — with Prolog Reasoning",
    description=(
        "Upload an image → BLIP generates a caption → "
        "spaCy extracts facts → SWI-Prolog runs inference rules. "
        "Returns caption + structured knowledge + symbolic inferences."
    ),
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# GLOBALS
# ============================================================

processor   = None
model       = None
nlp_model   = None
prolog      = None
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH  = "./blip_finetuned"
FALLBACK    = "Salesforce/blip-image-captioning-base"
RULES_FILE  = "knowledge_base.pl"


# ============================================================
# KNOWLEDGE BASE — written to disk on startup
# ============================================================

KNOWLEDGE_BASE_PL = """\
% ============================================================
% knowledge_base.pl — Image Captioning Inference Rules
% ============================================================

:- dynamic object/2.
:- dynamic action/2.
:- dynamic attribute/3.
:- dynamic scene/2.
:- dynamic relation/4.
:- dynamic caption/2.

% --- Core inference rules ---

active_image(Id) :-
    action(Id, _).

natural_scene(Id) :-
    scene(Id, outdoor) ; scene(Id, nature).

contains_animal(Id) :-
    Animals = [dog,cat,bird,horse,fish,elephant,lion,tiger,rabbit,cow,bear,deer,sheep,goat],
    member(A, Animals),
    object(Id, A).

crowded(Id) :-
    findall(O, object(Id, O), Objs),
    length(Objs, N), N > 4.

% Emotion/mood inferences from adjectives
happy_scene(Id) :-
    HappyWords = [happy,smiling,laughing,sunny,bright,colorful,playful,joyful],
    member(W, HappyWords),
    attribute(Id, _, W).

calm_scene(Id) :-
    CalmWords = [calm,quiet,peaceful,serene,still,empty,lone],
    member(W, CalmWords),
    attribute(Id, _, W).

% Image category inference
image_category(Id, animal_scene)     :- contains_animal(Id), natural_scene(Id).
image_category(Id, urban_action)     :- active_image(Id), scene(Id, urban).
image_category(Id, indoor_activity)  :- active_image(Id), scene(Id, indoor).
image_category(Id, outdoor_activity) :- active_image(Id), scene(Id, outdoor).
image_category(Id, nature_scene)     :- natural_scene(Id), \\+ active_image(Id).
image_category(Id, general)          :- \\+ contains_animal(Id), \\+ active_image(Id).

% Complexity score: count total facts for this image
complexity_score(Id, Score) :-
    findall(O, object(Id, O), Objs),
    findall(A, action(Id, A), Acts),
    findall(R, relation(Id, R, _, _), Rels),
    length(Objs, NO), length(Acts, NA), length(Rels, NR),
    Score is NO + NA + (NR * 2).
"""


# ============================================================
# STARTUP — load models and Prolog engine
# ============================================================

@app.on_event("startup")
async def startup():
    global processor, model, nlp_model, prolog

    # Write knowledge base to disk
    with open(RULES_FILE, "w") as f:
        f.write(KNOWLEDGE_BASE_PL)
    print(f"Knowledge base written: {RULES_FILE}")

    # Load BLIP
    model_to_load = MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK
    print(f"Loading BLIP from: {model_to_load}  |  device: {DEVICE}")
    processor = BlipProcessor.from_pretrained(model_to_load)
    model     = BlipForConditionalGeneration.from_pretrained(model_to_load)
    model     = model.to(DEVICE)
    model.eval()
    print("BLIP ready.")

    # Load spaCy
    print("Loading spaCy...")
    nlp_model = spacy.load("en_core_web_sm")
    print("spaCy ready.")

    # Load SWI-Prolog via pyswip
    try:
        from pyswip import Prolog as SWIProlog
        prolog = SWIProlog()
        prolog.consult(RULES_FILE)
        print("SWI-Prolog ready.")
    except Exception as e:
        print(f"WARNING: SWI-Prolog not available: {e}")
        print("Prolog inference will be skipped. Install swi-prolog + pyswip to enable.")
        prolog = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_atom(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9_ ]", "", s)
    s = s.replace(" ", "_")
    return s if s else "unknown"


def generate_caption(image: Image.Image) -> str:
    inputs = processor(image.convert("RGB"), return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs, 
            max_new_tokens=50, 
            do_sample=True, 
            top_p=0.9, 
            num_return_sequences=1
        )
    return processor.decode(out[0], skip_special_tokens=True)


def extract_facts(caption: str, image_id: str) -> dict:
    doc        = nlp_model(caption)
    objects    = list({t.text.lower() for t in doc if t.pos_ in ("NOUN","PROPN")})
    actions    = list({t.lemma_.lower() for t in doc if t.pos_ == "VERB"})
    attributes = list({t.text.lower() for t in doc if t.pos_ == "ADJ"})
    text       = caption.lower()
    scene_map  = {
        "outdoor": ["outdoor","park","street","road","sky","tree","beach","mountain","field"],
        "indoor":  ["indoor","room","kitchen","office","living","bedroom","table","sofa"],
        "urban":   ["city","building","car","bus","traffic","sidewalk"],
        "nature":  ["forest","grass","river","ocean","flower","lake"]
    }
    scene = "unknown"
    for s, kws in scene_map.items():
        if any(k in text for k in kws): scene = s; break
    relations = []
    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            for child in token.head.children:
                if child.dep_ in ("dobj","attr","prep"):
                    relations.append((token.text.lower(), token.head.lemma_.lower(), child.text.lower()))
    return {
        "image_id": image_id, "caption": caption,
        "objects": objects, "actions": actions,
        "attributes": attributes, "scene": scene, "relations": relations
    }


def assert_facts_to_prolog(facts: dict):
    """Push all extracted facts into SWI-Prolog as dynamic clauses."""
    img = facts["image_id"]
    safe_cap = facts["caption"].replace("'", "\\'")
    prolog.assertz(f"caption('{img}', '{safe_cap}')")
    prolog.assertz(f"scene('{img}', {safe_atom(facts['scene'])})")
    for obj  in facts["objects"]:    prolog.assertz(f"object('{img}', {safe_atom(obj)})")
    for act  in facts["actions"]:    prolog.assertz(f"action('{img}', {safe_atom(act)})")
    for attr in facts["attributes"]: prolog.assertz(f"attribute('{img}', general, {safe_atom(attr)})")
    for s, v, o in facts["relations"]:
        prolog.assertz(f"relation('{img}', {safe_atom(s)}, {safe_atom(v)}, {safe_atom(o)})")


def query_prolog(image_id: str) -> dict:
    """Run all inference rules and return a structured results dict."""
    img = image_id
    return {
        "objects":    [s["O"] for s in prolog.query(f"object('{img}', O)")],
        "actions":    [s["A"] for s in prolog.query(f"action('{img}', A)")],
        "attributes": list({s["At"] for s in prolog.query(f"attribute('{img}', _, At)")}),
        "scene":      next((s["S"] for s in prolog.query(f"scene('{img}', S)")), "unknown"),
        "relations":  [{"subject": s["Su"], "verb": s["V"], "object": s["O"]}
                       for s in prolog.query(f"relation('{img}', Su, V, O)")],
        "inferences": {
            "category":    next((s["C"] for s in prolog.query(f"image_category('{img}', C)")), "general"),
            "is_active":   bool(list(prolog.query(f"active_image('{img}')"))),
            "is_natural":  bool(list(prolog.query(f"natural_scene('{img}')"))),
            "has_animal":  bool(list(prolog.query(f"contains_animal('{img}')"))),
            "is_crowded":  bool(list(prolog.query(f"crowded('{img}')"))),
            "is_happy":    bool(list(prolog.query(f"happy_scene('{img}')"))),
            "is_calm":     bool(list(prolog.query(f"calm_scene('{img}')"))),
            "complexity":  next((int(s["S"]) for s in prolog.query(f"complexity_score('{img}', S)")), 0),
        }
    }


def retract_image(image_id: str):
    """Clean up all dynamic facts for a given image after processing."""
    img = image_id
    for pred in ["object","action","attribute","scene","relation","caption"]:
        list(prolog.query(f"retractall({pred}('{img}', _))"))


def build_prolog_text(facts: dict) -> str:
    """Generate a downloadable .pl file string for the image."""
    img   = facts["image_id"]
    lines = [
        f"% Generated by AI Image Captioning API",
        f"% Image ID: {img}",
        f"% Caption: {facts['caption']}",
        "",
        f"caption('{img}', '{facts['caption'].replace(chr(39), chr(92)+chr(39))}').",
        f"scene('{img}', {safe_atom(facts['scene'])}).",
    ]
    for obj  in facts["objects"]:    lines.append(f"object('{img}', {safe_atom(obj)}).")
    for act  in facts["actions"]:    lines.append(f"action('{img}', {safe_atom(act)}).")
    for attr in facts["attributes"]: lines.append(f"attribute('{img}', general, {safe_atom(attr)}).")
    for s,v,o in facts["relations"]: lines.append(f"relation('{img}', {safe_atom(s)}, {safe_atom(v)}, {safe_atom(o)}).")
    return "\n".join(lines)


# ============================================================
# CORE PIPELINE — used by both /caption and /caption-url
# ============================================================

def run_pipeline(image: Image.Image, filename: str = "upload") -> dict:
    image_id    = str(uuid.uuid4())[:8]
    start       = time.time()

    # 1. Generate caption with BLIP
    caption     = generate_caption(image)

    # 2. Extract structured facts with spaCy NLP
    facts       = extract_facts(caption, f"img_{image_id}")
    facts["image_id"] = f"img_{image_id}"

    # 3. Assert into Prolog and run inference (if SWI-Prolog available)
    prolog_result = None
    if prolog:
        try:
            assert_facts_to_prolog(facts)
            prolog_result = query_prolog(f"img_{image_id}")
            retract_image(f"img_{image_id}")
        except Exception as e:
            prolog_result = {"error": str(e)}

    elapsed = round(time.time() - start, 3)

    return {
        "image_id":         image_id,
        "filename":         filename,
        "image_size":       f"{image.width}x{image.height}",
        "caption":          caption,
        "inference_time_seconds": elapsed,
        "extracted_facts":  {
            "objects":    facts["objects"],
            "actions":    facts["actions"],
            "attributes": facts["attributes"],
            "scene":      facts["scene"],
            "relations":  [{"subject":s,"verb":v,"object":o} for s,v,o in facts["relations"]],
        },
        "prolog_inference": prolog_result,
        "prolog_file":      build_prolog_text(facts),   # downloadable .pl content
    }


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    return {
        "message": "AI Image Captioning API with Prolog Reasoning",
        "docs":    "http://localhost:8000/docs",
        "device":  DEVICE,
        "prolog":  prolog is not None
    }


@app.get("/health")
async def health():
    if model is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {
        "status":        "healthy",
        "blip_loaded":   True,
        "spacy_loaded":  nlp_model is not None,
        "prolog_loaded": prolog is not None,
        "device":        DEVICE
    }


@app.post("/caption")
async def caption_endpoint(file: UploadFile = File(...)):
    """
    Upload an image file → get caption + Prolog reasoning output.

    Response includes:
    - caption          : the AI-generated caption string
    - extracted_facts  : objects, actions, attributes, scene, relations
    - prolog_inference : inferred category, flags, complexity score
    - prolog_file      : the .pl file content you can save and load in SWI-Prolog
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model still loading")

    allowed = ["image/jpeg","image/png","image/webp","image/jpg"]
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported type: {file.content_type}")

    file_bytes = await file.read()
    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 10 MB)")

    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return run_pipeline(image, filename=file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/caption-url")
async def caption_url_endpoint(url: str = Query(..., description="Public image URL")):
    """Pass a public image URL and get caption + Prolog output."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model still loading")
    import requests as req
    try:
        resp  = req.get(url, timeout=10, stream=True)
        resp.raise_for_status()
        image = Image.open(resp.raw).convert("RGB")
        result = run_pipeline(image, filename=url.split("/")[-1])
        result["source_url"] = url
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/prolog/custom-query")
async def custom_prolog_query(
    image_id: str = Query(..., description="Image ID returned from /caption"),
    facts_pl: str = Query(..., description="Prolog fact string (the prolog_file from /caption response)"),
    query:    str = Query(..., description="Prolog query to run, e.g. contains_animal(img_abc123)")
):
    """
    Assert an image's Prolog facts and run any custom query you provide.
    Useful for experimenting with your own rules.
    """
    if prolog is None:
        raise HTTPException(status_code=503, detail="SWI-Prolog not available")
    try:
        # Load user-provided facts into a temp file
        tmp = f"/tmp/custom_{image_id}.pl"
        with open(tmp, "w") as f:
            f.write(facts_pl)
        prolog.consult(tmp)
        results = list(prolog.query(query))
        os.remove(tmp)
        return {"query": query, "results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("step4_api:app", host="0.0.0.0", port=8000, reload=True)
