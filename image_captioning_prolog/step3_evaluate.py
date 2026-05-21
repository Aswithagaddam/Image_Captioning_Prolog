# ============================================================
# STEP 3 — Evaluate Model with BLEU + Prolog Inference Scoring
# ============================================================
# Evaluation has two parts:
#
#   PART A — Standard NLP metrics (same as original)
#     BLEU-1, BLEU-2, BLEU-3, BLEU-4
#     Compares generated captions to human reference captions
#
#   PART B — Prolog Semantic Scoring (NEW)
#     Loads facts into SWI-Prolog via pyswip
#     Runs symbolic inference rules on each image
#     Reports: scene accuracy, object coverage, relation extraction rate
#
# HOW TO RUN:
#   pip install nltk pyswip transformers torch Pillow spacy
#   python -m spacy download en_core_web_sm
#   apt-get install swi-prolog    (Linux / Colab)
#   python step3_evaluate.py
# ============================================================

# ----- CELL 1: Imports -----
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import json
import nltk
import spacy
import re
import os
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

nltk.download("punkt")
nlp = spacy.load("en_core_web_sm")

# ----- CELL 2: Load model -----
MODEL_PATH = "./blip_finetuned"
FALLBACK   = "Salesforce/blip-image-captioning-base"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

model_to_load = MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK
print(f"Loading: {model_to_load}  |  device: {DEVICE}")

processor = BlipProcessor.from_pretrained(model_to_load)
model     = BlipForConditionalGeneration.from_pretrained(model_to_load)
model     = model.to(DEVICE)
model.eval()

# ----- CELL 3: Helper functions (same as step1/step2) -----
def safe_atom(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9_ ]", "", s)
    s = s.replace(" ", "_")
    return s if s else "unknown"


def generate_caption(image: Image.Image) -> str:
    inputs = processor(image.convert("RGB"), return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50, num_beams=4, early_stopping=True)
    return processor.decode(out[0], skip_special_tokens=True)


def extract_facts(caption: str, image_id: str) -> dict:
    doc        = nlp(caption)
    objects    = list({t.text.lower() for t in doc if t.pos_ in ("NOUN", "PROPN")})
    actions    = list({t.lemma_.lower() for t in doc if t.pos_ == "VERB"})
    attributes = list({t.text.lower() for t in doc if t.pos_ == "ADJ"})
    text       = caption.lower()
    scene_map  = {
        "outdoor": ["outdoor","outside","park","street","road","sky","tree","beach","mountain","field"],
        "indoor":  ["indoor","inside","room","kitchen","office","living","bedroom","table","sofa"],
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


# ----- CELL 4: BLEU evaluation -----
def calculate_bleu(generated: list, references: list) -> dict:
    hypotheses = [c.lower().split() for c in generated]
    refs        = [[r.lower().split() for r in rs] for rs in references]
    sm = SmoothingFunction().method1
    return {
        "BLEU-1": round(corpus_bleu(refs, hypotheses, weights=(1,0,0,0),            smoothing_function=sm), 4),
        "BLEU-2": round(corpus_bleu(refs, hypotheses, weights=(.5,.5,0,0),          smoothing_function=sm), 4),
        "BLEU-3": round(corpus_bleu(refs, hypotheses, weights=(.33,.33,.33,0),      smoothing_function=sm), 4),
        "BLEU-4": round(corpus_bleu(refs, hypotheses, weights=(.25,.25,.25,.25),    smoothing_function=sm), 4),
    }


# ----- CELL 5: Prolog semantic evaluation -----
def evaluate_with_prolog(test_items: list) -> dict:
    """
    For each test image:
      1. Assert facts into SWI-Prolog via pyswip
      2. Run inference rules
      3. Collect results
      4. Retract (clean up)

    Returns aggregate semantic metrics.
    """
    try:
        from pyswip import Prolog
    except ImportError:
        print("pyswip not installed — skipping Prolog evaluation.")
        print("Install with: pip install pyswip  (and apt-get install swi-prolog)")
        return {}

    prolog = Prolog()

    # Write the rules file inline for evaluation
    rules_text = """
:- dynamic object/2.
:- dynamic action/2.
:- dynamic attribute/3.
:- dynamic scene/2.
:- dynamic relation/4.
:- dynamic caption/2.

active_image(Id) :- action(Id, _).
natural_scene(Id) :- scene(Id, outdoor) ; scene(Id, nature).
contains_animal(Id) :-
    Animals = [dog,cat,bird,horse,fish,elephant,lion,tiger,rabbit,cow,bear,deer],
    member(A, Animals), object(Id, A).
crowded(Id) :-
    findall(O, object(Id, O), Objs), length(Objs, N), N > 4.
image_category(Id, animal_scene)    :- contains_animal(Id), natural_scene(Id).
image_category(Id, urban_action)    :- active_image(Id), scene(Id, urban).
image_category(Id, indoor_activity) :- active_image(Id), scene(Id, indoor).
image_category(Id, general)         :-
    \\+ contains_animal(Id), \\+ active_image(Id).
"""
    with open("eval_rules.pl", "w") as f:
        f.write(rules_text)
    prolog.consult("eval_rules.pl")

    results = []

    for item in test_items:
        img    = item["image_id"]
        facts  = item["facts"]
        gt_objects = item.get("ground_truth_objects", [])  # if you have ground truth

        # Assert facts
        safe_cap = facts["caption"].replace("'", "\\'")
        prolog.assertz(f"caption('{img}', '{safe_cap}')")
        prolog.assertz(f"scene('{img}', {safe_atom(facts['scene'])})")
        for obj  in facts["objects"]:    prolog.assertz(f"object('{img}', {safe_atom(obj)})")
        for act  in facts["actions"]:    prolog.assertz(f"action('{img}', {safe_atom(act)})")
        for attr in facts["attributes"]: prolog.assertz(f"attribute('{img}', general, {safe_atom(attr)})")
        for s, v, o in facts["relations"]: prolog.assertz(f"relation('{img}', {safe_atom(s)}, {safe_atom(v)}, {safe_atom(o)})")

        # Run inference queries
        r = {
            "image_id":    img,
            "scene":       facts["scene"],
            "num_objects": len(facts["objects"]),
            "num_actions": len(facts["actions"]),
            "num_relations": len(facts["relations"]),
            "is_active":   bool(list(prolog.query(f"active_image('{img}')"))),
            "is_natural":  bool(list(prolog.query(f"natural_scene('{img}')"))),
            "has_animal":  bool(list(prolog.query(f"contains_animal('{img}')"))),
            "is_crowded":  bool(list(prolog.query(f"crowded('{img}')"))),
            "category":    next((s["C"] for s in prolog.query(f"image_category('{img}', C)")), "general"),
        }
        results.append(r)

        # Retract all facts for this image
        for pred in ["object","action","attribute","scene","relation","caption"]:
            list(prolog.query(f"retractall({pred}('{img}', _))"))

    # Aggregate metrics
    n = len(results)
    metrics = {
        "total_images":          n,
        "avg_objects_per_image": round(sum(r["num_objects"] for r in results) / max(n,1), 2),
        "avg_actions_per_image": round(sum(r["num_actions"] for r in results) / max(n,1), 2),
        "avg_relations_per_image": round(sum(r["num_relations"] for r in results) / max(n,1), 2),
        "pct_active_images":     round(sum(r["is_active"] for r in results)  / max(n,1) * 100, 1),
        "pct_natural_scenes":    round(sum(r["is_natural"] for r in results)  / max(n,1) * 100, 1),
        "pct_contains_animal":   round(sum(r["has_animal"] for r in results)  / max(n,1) * 100, 1),
        "pct_crowded":           round(sum(r["is_crowded"] for r in results)  / max(n,1) * 100, 1),
        "category_distribution": {},
        "per_image_results":     results
    }

    for r in results:
        cat = r["category"]
        metrics["category_distribution"][cat] = metrics["category_distribution"].get(cat, 0) + 1

    os.remove("eval_rules.pl")
    return metrics


# ----- CELL 6: Test dataset -----
test_data = [
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg",
        "references": [
            "a cat sitting on a white surface",
            "an orange cat looking at the camera",
            "a cute cat with green eyes",
            "a cat posing for a photo",
            "close up of an orange tabby cat"
        ],
        "ground_truth_objects": ["cat"]
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Dog_Breeds.jpg/320px-Dog_Breeds.jpg",
        "references": [
            "a group of dogs sitting together",
            "many different dog breeds",
            "dogs of various breeds",
            "several dogs posing for a picture",
            "a collection of different dog breeds"
        ],
        "ground_truth_objects": ["dog"]
    }
]

# ----- CELL 7: Run evaluation -----
generated_captions    = []
reference_captions    = []
prolog_eval_items     = []

print("Generating captions and extracting facts...")
print("=" * 60)

for i, item in enumerate(test_data):
    try:
        image   = Image.open(requests.get(item["url"], stream=True).raw).convert("RGB")
        caption = generate_caption(image)
        facts   = extract_facts(caption, f"eval_img_{i:03d}")

        generated_captions.append(caption)
        reference_captions.append(item["references"])
        prolog_eval_items.append({
            "image_id": f"eval_img_{i:03d}",
            "facts": facts,
            "ground_truth_objects": item.get("ground_truth_objects", [])
        })

        print(f"[{i+1}] {caption}")
        print(f"     Objects: {facts['objects']} | Scene: {facts['scene']}")
        print("-" * 60)
    except Exception as e:
        print(f"Error on image {i}: {e}")

# ----- CELL 8: BLEU scores -----
bleu_scores = {}
if generated_captions:
    bleu_scores = calculate_bleu(generated_captions, reference_captions)
    print("\n=== BLEU SCORES ===")
    for metric, score in bleu_scores.items():
        bar = "█" * int(score * 30)
        print(f"{metric}: {score:.4f}  {bar}")

# ----- CELL 9: Prolog semantic scores -----
print("\n=== PROLOG SEMANTIC EVALUATION ===")
prolog_metrics = evaluate_with_prolog(prolog_eval_items)

if prolog_metrics:
    print(f"Avg objects per image:    {prolog_metrics['avg_objects_per_image']}")
    print(f"Avg actions per image:    {prolog_metrics['avg_actions_per_image']}")
    print(f"Avg relations per image:  {prolog_metrics['avg_relations_per_image']}")
    print(f"% images with actions:    {prolog_metrics['pct_active_images']}%")
    print(f"% natural/outdoor scenes: {prolog_metrics['pct_natural_scenes']}%")
    print(f"% images with animals:    {prolog_metrics['pct_contains_animal']}%")
    print(f"Category distribution:    {prolog_metrics['category_distribution']}")

# ----- CELL 10: Save combined results -----
results = {
    "model_path":       model_to_load,
    "num_test_images":  len(generated_captions),
    "bleu_scores":      bleu_scores,
    "prolog_metrics":   prolog_metrics,
    "examples": [
        {"generated": gen, "references": refs}
        for gen, refs in zip(generated_captions, reference_captions)
    ]
}

with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to evaluation_results.json")
