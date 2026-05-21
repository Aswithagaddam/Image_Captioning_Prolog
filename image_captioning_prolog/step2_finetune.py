# ============================================================
# STEP 2 — Fine-tune BLIP + Build Prolog Knowledge Base
# ============================================================
# Uses 'conceptual_captions' dataset — works on all machines,
# no GPU required, no loading script issues.
#
# On MacBook Air (CPU): set max_samples=200, num_epochs=1
# On Google Colab GPU: set max_samples=2000, num_epochs=3
# ============================================================
 
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import spacy
import os
import re
import json
import shutil
import matplotlib
matplotlib.use("Agg")           # no display needed on Mac
import matplotlib.pyplot as plt
 
# ============================================================
# CONFIGURATION — change these for your machine
# ============================================================
CONFIG = {
    "model_name":    "Salesforce/blip-image-captioning-base",
    "batch_size":    2,        # 2 is safe for CPU / MacBook Air
    "num_epochs":    1,        # 1 epoch is enough for a quick test on CPU
    "learning_rate": 5e-5,
    "max_samples":   200,      # 200 = ~10 min on CPU | 2000 = ~40 min on GPU
    "max_length":    50,
    "save_path":     "./blip_finetuned",
    "prolog_kb_dir": "./prolog_kb",
    "device":        "cuda" if torch.cuda.is_available() else "cpu"
}
 
os.makedirs(CONFIG["save_path"],     exist_ok=True)
os.makedirs(CONFIG["prolog_kb_dir"], exist_ok=True)
 
print(f"Device : {CONFIG['device']}")
print(f"Training {CONFIG['max_samples']} samples for {CONFIG['num_epochs']} epoch(s)")
 
# ============================================================
# LOAD spaCy
# ============================================================
print("Loading spaCy...")
nlp = spacy.load("en_core_web_sm")
print("spaCy ready.")
 
# ============================================================
# NLP HELPERS
# ============================================================
def safe_atom(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9_ ]", "", s)
    s = s.replace(" ", "_")
    return s if s else "unknown"
 
 
def extract_facts(caption: str, image_id: str) -> dict:
    doc        = nlp(caption)
    objects    = list({t.text.lower() for t in doc if t.pos_ in ("NOUN", "PROPN")})
    actions    = list({t.lemma_.lower() for t in doc if t.pos_ == "VERB"})
    attributes = list({t.text.lower() for t in doc if t.pos_ == "ADJ"})
    text       = caption.lower()
    scene_map  = {
        "outdoor": ["outdoor","park","street","road","sky","tree","beach","mountain","field","outside"],
        "indoor":  ["indoor","room","kitchen","office","living","bedroom","table","sofa","inside"],
        "urban":   ["city","building","car","bus","traffic","sidewalk","downtown"],
        "nature":  ["forest","grass","river","ocean","flower","lake","waterfall"]
    }
    scene = "unknown"
    for s, kws in scene_map.items():
        if any(k in text for k in kws):
            scene = s
            break
    relations = []
    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            for child in token.head.children:
                if child.dep_ in ("dobj", "attr", "prep"):
                    relations.append((token.text.lower(), token.head.lemma_.lower(), child.text.lower()))
    return {"image_id": image_id, "caption": caption, "objects": objects,
            "actions": actions, "attributes": attributes, "scene": scene, "relations": relations}
 
 
def facts_to_prolog_lines(facts: dict) -> list:
    img      = facts["image_id"]
    safe_cap = facts["caption"].replace("'", "\\'")
    lines    = [f"% {img} — {facts['caption']}",
                f"caption('{img}', '{safe_cap}').",
                f"scene('{img}', {safe_atom(facts['scene'])})."]
    for obj  in facts["objects"]:    lines.append(f"object('{img}', {safe_atom(obj)}).")
    for act  in facts["actions"]:    lines.append(f"action('{img}', {safe_atom(act)}).")
    for attr in facts["attributes"]: lines.append(f"attribute('{img}', general, {safe_atom(attr)}).")
    for s,v,o in facts["relations"]: lines.append(f"relation('{img}', {safe_atom(s)}, {safe_atom(v)}, {safe_atom(o)}).")
    return lines
 
# ============================================================
# LOAD DATASET
# ============================================================
# conceptual_captions = 3M web image-caption pairs, clean Parquet,
# no loading script, works everywhere.
# Each sample has: 'image_url' (string) and 'caption' (string)
# We download images on the fly — no huge local download needed.
 
print("\nLoading Conceptual Captions dataset (streaming)...")
raw_dataset = load_dataset(
    "google-research-datasets/conceptual_captions",
    split="train",
    streaming=True
)
CAPTION_KEY = "caption"
print("Dataset ready.")
 
# ============================================================
# COLLECT SAMPLES
# Download images from URLs on the fly. Skip broken URLs.
# ============================================================
print(f"Collecting {CONFIG['max_samples']} samples (downloading images)...")
 
data_samples = []
skipped      = 0
iterator     = iter(raw_dataset)
 
while len(data_samples) < CONFIG["max_samples"]:
    try:
        sample = next(iterator)
    except StopIteration:
        break
 
    url     = sample.get("image_url", "")
    caption = sample.get(CAPTION_KEY, "").strip()
 
    if not url or not caption:
        skipped += 1
        continue
 
    try:
        resp = requests.get(url, timeout=5, stream=True)
        resp.raise_for_status()
        image = Image.open(BytesIO(resp.content)).convert("RGB")
        # store PIL image + caption together
        data_samples.append({"image": image, "caption": caption})
    except Exception:
        skipped += 1
        continue
 
    if len(data_samples) % 50 == 0 and len(data_samples) > 0:
        print(f"  Collected {len(data_samples)} samples (skipped {skipped})...")
 
print(f"Done. Collected: {len(data_samples)} | Skipped (broken URLs): {skipped}")
print(f"Example caption: {data_samples[0]['caption']}")
 
# ============================================================
# DATASET CLASS
# ============================================================
class CaptionDataset(Dataset):
    def __init__(self, samples, processor, max_length=50):
        self.samples    = samples
        self.processor  = processor
        self.max_length = max_length
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        sample  = self.samples[idx]
        image   = sample["image"]
        caption = sample["caption"]
 
        try:
            encoding = self.processor(
                images=image,
                text=caption,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
        except Exception:
            return None
 
        return {
            "pixel_values":  encoding["pixel_values"].squeeze(),
            "input_ids":     encoding["input_ids"].squeeze(),
            "attention_mask":encoding["attention_mask"].squeeze(),
            "caption":       caption,
            "image_id":      f"cc_{idx:06d}"
        }
 
 
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {
        "pixel_values":  torch.stack([b["pixel_values"]  for b in batch]),
        "input_ids":     torch.stack([b["input_ids"]     for b in batch]),
        "attention_mask":torch.stack([b["attention_mask"] for b in batch]),
        "captions":      [b["caption"]  for b in batch],
        "image_ids":     [b["image_id"] for b in batch],
    }
 
# ============================================================
# LOAD MODEL
# ============================================================
print("\nLoading BLIP model...")
processor = BlipProcessor.from_pretrained(CONFIG["model_name"])
model     = BlipForConditionalGeneration.from_pretrained(CONFIG["model_name"])
model     = model.to(CONFIG["device"])
print("BLIP ready.")
 
split_idx     = int(0.9 * len(data_samples))
train_samples = data_samples[:split_idx]
val_samples   = data_samples[split_idx:]
 
train_ds = CaptionDataset(train_samples, processor, CONFIG["max_length"])
val_ds   = CaptionDataset(val_samples,   processor, CONFIG["max_length"])
 
train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,  collate_fn=collate_fn, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=0)
 
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
 
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
 
# ============================================================
# TRAINING LOOP + PROLOG KB BUILDING
# ============================================================
print("\nStarting training...")
print("=" * 60)
 
best_val_loss = float("inf")
history       = {"train_loss": [], "val_loss": []}
 
for epoch in range(CONFIG["num_epochs"]):
    print(f"\nEpoch {epoch + 1}/{CONFIG['num_epochs']}")
 
    # ---- Train ----
    model.train()
    total_train, n_train = 0, 0
    for batch in tqdm(train_loader, desc="Training"):
        if batch is None:
            continue
        pv = batch["pixel_values"].to(CONFIG["device"])
        ii = batch["input_ids"].to(CONFIG["device"])
        am = batch["attention_mask"].to(CONFIG["device"])
 
        outputs = model(pixel_values=pv, input_ids=ii, attention_mask=am, labels=ii)
        loss    = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train += loss.item()
        n_train     += 1
 
    avg_train = total_train / max(n_train, 1)
    history["train_loss"].append(avg_train)
    print(f"  Train Loss: {avg_train:.4f}")
 
    # ---- Validate + Build Prolog KB ----
    model.eval()
    total_val, n_val = 0, 0
    prolog_lines = [
        "% ============================================================",
        f"% Prolog KB — epoch {epoch + 1} validation captions",
        "% ============================================================",
        "",
        ":- dynamic object/2.",
        ":- dynamic action/2.",
        ":- dynamic attribute/3.",
        ":- dynamic scene/2.",
        ":- dynamic relation/4.",
        ":- dynamic caption/2.",
        ""
    ]
 
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating + Prolog KB"):
            if batch is None:
                continue
            pv = batch["pixel_values"].to(CONFIG["device"])
            ii = batch["input_ids"].to(CONFIG["device"])
            am = batch["attention_mask"].to(CONFIG["device"])
 
            outputs   = model(pixel_values=pv, input_ids=ii, attention_mask=am, labels=ii)
            total_val += outputs.loss.item()
            n_val     += 1
 
            for caption, image_id in zip(batch["captions"], batch["image_ids"]):
                facts = extract_facts(caption, image_id)
                prolog_lines.extend(facts_to_prolog_lines(facts))
                prolog_lines.append("")
 
    avg_val = total_val / max(n_val, 1)
    history["val_loss"].append(avg_val)
    print(f"  Val Loss:   {avg_val:.4f}")
 
    kb_path = os.path.join(CONFIG["prolog_kb_dir"], f"val_facts_epoch{epoch + 1}.pl")
    with open(kb_path, "w") as f:
        f.write("\n".join(prolog_lines))
    print(f"  Prolog KB:  {kb_path}  ({len(prolog_lines)} lines)")
 
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        model.save_pretrained(CONFIG["save_path"])
        processor.save_pretrained(CONFIG["save_path"])
        print(f"  Model saved -> {CONFIG['save_path']}")
 
# ============================================================
# SAVE BEST KB + HISTORY
# ============================================================
best_epoch = history["val_loss"].index(min(history["val_loss"])) + 1
best_kb    = os.path.join(CONFIG["prolog_kb_dir"], f"val_facts_epoch{best_epoch}.pl")
canonical  = os.path.join(CONFIG["prolog_kb_dir"], "val_facts.pl")
shutil.copy(best_kb, canonical)
 
print(f"\nBest epoch: {best_epoch}  |  Best val loss: {best_val_loss:.4f}")
print(f"Canonical Prolog KB: {canonical}")
 
with open("training_history.json", "w") as f:
    json.dump(history, f, indent=2)
print("Training history saved.")
 
# ============================================================
# PLOT
# ============================================================
plt.figure(figsize=(8, 4))
plt.plot(history["train_loss"], label="Train Loss", marker="o")
plt.plot(history["val_loss"],   label="Val Loss",   marker="s")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("BLIP Fine-tuning Progress")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("training_plot.png")
print("Plot saved as training_plot.png")
 
# ============================================================
# TEST FINE-TUNED MODEL
# ============================================================
print("\nTesting fine-tuned model...")
ft_processor = BlipProcessor.from_pretrained(CONFIG["save_path"])
ft_model     = BlipForConditionalGeneration.from_pretrained(CONFIG["save_path"])
ft_model     = ft_model.to(CONFIG["device"])
ft_model.eval()
 
 
def caption_with_finetuned(url: str) -> str:
    resp  = requests.get(url, timeout=10)
    image = Image.open(BytesIO(resp.content)).convert("RGB")
    inputs = ft_processor(image, return_tensors="pt").to(CONFIG["device"])
    with torch.no_grad():
        out = ft_model.generate(**inputs, max_new_tokens=50, num_beams=4, early_stopping=True)
    return ft_processor.decode(out[0], skip_special_tokens=True)
 
 
test_url = "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=320"
print(f"Test caption: {caption_with_finetuned(test_url)}")
 
# ============================================================
# PROLOG KB PREVIEW
# ============================================================
print(f"\nFirst 20 lines of Prolog KB ({canonical}):")
with open(canonical) as f:
    for i, line in enumerate(f):
        if i >= 20: break
        print(line, end="")
 
print("\n\nStep 2 complete!")
print(f"  Fine-tuned model saved : {CONFIG['save_path']}/")
print(f"  Prolog knowledge base  : {canonical}")
print(f"  Training plot          : training_plot.png")
 