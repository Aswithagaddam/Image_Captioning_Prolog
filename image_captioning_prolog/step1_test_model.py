# ============================================================
# STEP 1 — Test BLIP model + Generate Prolog Facts
# ============================================================
# HOW TO USE:
#   1. Go to colab.research.google.com
#   2. Create a new notebook
#   3. Paste this code and run cell by cell
#   4. Enable GPU: Runtime > Change runtime type > GPU
#
# NEW vs original: After captioning, we use spaCy to extract
# facts and write them as Prolog predicates into a .pl file.
# ============================================================

# ----- CELL 1: Install libraries -----
# !pip install transformers torch torchvision Pillow requests spacy
# !python -m spacy download en_core_web_sm
# For Prolog on Colab: !apt-get install -y swi-prolog && pip install pyswip

# ----- CELL 2: Imports -----
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import torch
import spacy
import re
import os

# ----- CELL 3: Load BLIP model -----
print("Loading BLIP model...")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device    = "cuda" if torch.cuda.is_available() else "cpu"
model     = model.to(device)
model.eval()

print(f"BLIP ready on: {device}")

# ----- CELL 4: Load spaCy NLP model -----
# spaCy is used to parse the caption text into structured facts
print("Loading spaCy NLP model...")
nlp = spacy.load("en_core_web_sm")
print("spaCy ready!")

# ----- CELL 5: Caption generation function -----
def generate_caption(image: Image.Image) -> str:
    """
    Run BLIP on a PIL Image and return the caption string.
    """
    inputs = processor(image.convert("RGB"), return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=4,
            early_stopping=True
        )

    return processor.decode(output[0], skip_special_tokens=True)


# ----- CELL 6: NLP fact extraction -----
def extract_facts(caption: str, image_id: str = "img1") -> dict:
    """
    Parse the caption with spaCy and extract structured facts:
      - objects  (nouns / proper nouns)
      - actions  (verbs, lemmatized)
      - attributes (adjectives)
      - scene    (detected from keyword lists)
      - relations (subject-verb-object triples)

    Returns a dict of lists.
    """
    doc = nlp(caption)

    objects    = list({t.text.lower() for t in doc if t.pos_ in ("NOUN", "PROPN")})
    actions    = list({t.lemma_.lower() for t in doc if t.pos_ == "VERB"})
    attributes = list({t.text.lower() for t in doc if t.pos_ == "ADJ"})

    # Simple scene detection
    text = caption.lower()
    scene_map = {
        "outdoor": ["outdoor","outside","park","street","road","sky","tree","beach","mountain","field"],
        "indoor":  ["indoor","inside","room","kitchen","office","living","bedroom","table","sofa"],
        "urban":   ["city","building","car","bus","traffic","sidewalk","downtown"],
        "nature":  ["forest","grass","river","ocean","flower","lake","waterfall"]
    }
    scene = "unknown"
    for s, keywords in scene_map.items():
        if any(k in text for k in keywords):
            scene = s
            break

    # Subject-verb-object triples
    relations = []
    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            for child in token.head.children:
                if child.dep_ in ("dobj", "attr", "prep"):
                    relations.append((token.text.lower(), token.head.lemma_.lower(), child.text.lower()))

    return {
        "image_id": image_id,
        "caption":  caption,
        "objects":  objects,
        "actions":  actions,
        "attributes": attributes,
        "scene":    scene,
        "relations": relations
    }


# ----- CELL 7: Prolog fact writer -----
def facts_to_prolog(facts: dict, output_path: str = "image_facts.pl") -> str:
    """
    Convert extracted facts dict into a Prolog .pl file.
    Each piece of information becomes a Prolog predicate (fact).

    Prolog syntax reminder:
      predicate_name(argument1, argument2).   <- ends with a period
      Atoms must be lowercase or quoted.
    """
    img = facts["image_id"]

    def safe_atom(s: str) -> str:
        """Make a string safe as a Prolog atom — lowercase, underscored, no special chars."""
        s = s.lower().strip()
        s = re.sub(r"[^a-z0-9_ ]", "", s)   # strip special chars
        s = s.replace(" ", "_")
        return s if s else "unknown"

    lines = []
    lines.append(f"% ============================================================")
    lines.append(f"% Prolog facts generated from image: {img}")
    lines.append(f"% Caption: {facts['caption']}")
    lines.append(f"% ============================================================")
    lines.append("")

    # caption/2 — the full caption text
    safe_cap = facts["caption"].replace("'", "\\'")
    lines.append(f"caption('{img}', '{safe_cap}').")
    lines.append("")

    # scene/2 — detected scene type
    lines.append(f"scene('{img}', {safe_atom(facts['scene'])}).")
    lines.append("")

    # object/2 — each detected object noun
    lines.append("% Detected objects (nouns from caption)")
    for obj in facts["objects"]:
        lines.append(f"object('{img}', {safe_atom(obj)}).")
    lines.append("")

    # action/2 — each detected verb
    lines.append("% Detected actions (verbs from caption)")
    for act in facts["actions"]:
        lines.append(f"action('{img}', {safe_atom(act)}).")
    lines.append("")

    # attribute/3 — each adjective, linked generically
    lines.append("% Detected attributes (adjectives from caption)")
    for attr in facts["attributes"]:
        lines.append(f"attribute('{img}', general, {safe_atom(attr)}).")
    lines.append("")

    # relation/4 — subject-verb-object triples
    if facts["relations"]:
        lines.append("% Detected relations (subject-verb-object triples)")
        for subj, verb, obj in facts["relations"]:
            lines.append(f"relation('{img}', {safe_atom(subj)}, {safe_atom(verb)}, {safe_atom(obj)}).")
        lines.append("")

    prolog_text = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(prolog_text)

    print(f"Prolog facts written to: {output_path}")
    return prolog_text


# ----- CELL 8: Run the full pipeline on a test image -----
image_url = "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=320"

print("Downloading test image...")
image = Image.open(requests.get(image_url, stream=True).raw)

print("Generating caption...")
caption = generate_caption(image)
print(f"\nCaption: {caption}")

print("\nExtracting facts with spaCy...")
facts = extract_facts(caption, image_id="test_img_01")
print(f"Objects:     {facts['objects']}")
print(f"Actions:     {facts['actions']}")
print(f"Attributes:  {facts['attributes']}")
print(f"Scene:       {facts['scene']}")
print(f"Relations:   {facts['relations']}")

print("\nWriting Prolog facts file...")
prolog_text = facts_to_prolog(facts, "image_facts.pl")

print("\n--- Generated Prolog file contents ---")
print(prolog_text)

# ----- CELL 9: Test with your own uploaded image -----
# Uncomment to use on Colab with a file you upload manually

# from google.colab import files
# uploaded = files.upload()
# filename = list(uploaded.keys())[0]
# image = Image.open(filename)
# caption = generate_caption(image)
# facts   = extract_facts(caption, image_id="my_image")
# prolog_text = facts_to_prolog(facts, f"{filename}.pl")
# print(prolog_text)
