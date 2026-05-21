# AI Image Captioning with Prolog Symbolic Reasoning

An end-to-end multimodal pipeline that integrates **Deep Learning (Computer Vision + NLP)** with **Symbolic Logic (Prolog)**. The project generates captions for images, extracts syntactic and semantic structures, and performs logical reasoning to infer higher-level concepts, categories, moods, and complexity.

---

## ⚙️ System Architecture & Workflow

```
┌─────────┐      ┌──────────────┐      ┌─────────┐
│  Image  │ ───> │  BLIP Model  │ ───> │ Caption │
└─────────┘      └──────────────┘      └─────────┘
                                            │
                                            ▼
┌─────────┐      ┌──────────────┐      ┌─────────┐
│ Symbolic│ ───> │  SWI-Prolog  │ <─── │  spaCy  │
│ Results │      │ Inference KB │      │   NLP   │
└─────────┘      └──────────────┘      └─────────┘
```

1. **Vision (BLIP)**: Salesforce's BLIP (Bootstrapped Language-Image Pre-training) model analyzes an input image and generates a descriptive natural language caption.
2. **NLP Fact Extraction (spaCy)**: The generated caption is parsed into syntactic structures (nouns as objects, verbs as actions, adjectives as attributes, and subject-verb-object relationships).
3. **Symbolic Reasoning (SWI-Prolog)**: These structured facts are dynamically asserted into a Prolog knowledge base (`knowledge_base.pl`). Logic rules run queries to infer metadata like scene mood, activity level, image category, and overall scene complexity.

---

## 📂 Project Structure

```bash
Image_Captioning_Prolog/
├── image_captioning_prolog/
│   ├── prolog_kb/                # Validation facts output folder
│   │   ├── val_facts.pl          # Combined/best validation facts
│   │   └── val_facts_epoch1.pl   # Epoch-specific validation facts
│   ├── step1_test_model.py       # Test base BLIP model & output initial facts (.pl)
│   ├── step2_finetune.py         # Fine-tune BLIP on Conceptual Captions + build KB
│   ├── step3_evaluate.py         # Evaluate captions using BLEU metrics
│   ├── step4_api.py              # FastAPI server connecting Python & PySwip (Prolog)
│   ├── step5_frontend.html       # Glassmorphism HTML/JS dashboard interface
│   ├── requirements.txt          # Python dependencies
│   ├── knowledge_base.pl         # Static Prolog logic rules
│   ├── image_facts.pl            # Last generated Prolog facts file
│   ├── training_history.json     # Fine-tuning loss log
│   └── training_plot.png         # Epoch vs Loss visualization
├── .gitignore                    # Git file exclusions
└── README.md                     # Documentation
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites (SWI-Prolog)
For the Prolog engine (`pyswip`) to function, **SWI-Prolog** must be installed on your operating system:
* **Windows**: Download the installer from the [SWI-Prolog Stable Download page](https://www.swi-prolog.org/download/stable) and make sure to add it to your System PATH during installation.
* **macOS**: `brew install swi-prolog`
* **Linux (Ubuntu/Debian)**: `sudo apt-get install -y swi-prolog`

### 2. Install Python Dependencies
Create a virtual environment, activate it, and install the required libraries:
```bash
# Create and activate venv
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # macOS/Linux

# Install requirements
pip install -r image_captioning_prolog/requirements.txt

# Download spaCy English language model
python -m spacy download en_core_web_sm
```

---

## 🚀 Usage Guide

### Step 1: Run a Test Inference
Generates a caption for a sample internet image, parses it, and writes the Prolog representation to `image_facts.pl`:
```bash
python image_captioning_prolog/step1_test_model.py
```

### Step 2: Fine-Tuning the Model
Fine-tunes the BLIP model on a streaming subset of Google's `conceptual_captions` dataset, saves the weights to `blip_finetuned/`, and exports validation facts:
```bash
python image_captioning_prolog/step2_finetune.py
```

### Step 3: Performance Evaluation
Evaluates the fine-tuned model against validation references using BLEU scores:
```bash
python image_captioning_prolog/step3_evaluate.py
```

### Step 4: Launch the API Server
Start the FastAPI backend. It loads the BLIP model and interfaces directly with the local SWI-Prolog runtime:
```bash
python image_captioning_prolog/step4_api.py
```
* **Interactive Docs**: Once running, visit [http://localhost:8000/docs](http://localhost:8000/docs) for the Swagger API UI.

### Step 5: Open the Dashboard Frontend
Open `image_captioning_prolog/step5_frontend.html` directly in your browser (or serve it via any static local server). 
* Drag and drop files or select images to watch the pipeline execute in real-time.
* View generated captions, structural facts, raw `.pl` code, and logical inferences in their respective tabs.

---

## 🧩 Prolog Inferences Examples

The system uses rules inside `knowledge_base.pl` to deduce new knowledge:

```prolog
% Deduces if the image contains animals
contains_animal(Id) :-
    Animals = [dog,cat,bird,horse,fish,elephant,lion,tiger,cow,bear],
    member(A, Animals),
    object(Id, A).

% Deduces the category of scene based on location and activity
image_category(Id, animal_scene)     :- contains_animal(Id), natural_scene(Id).
image_category(Id, indoor_activity)  :- active_image(Id), scene(Id, indoor).
image_category(Id, outdoor_activity) :- active_image(Id), scene(Id, outdoor).
```