===================================================
  LIME Explainer — Quick Start
===================================================

Requirements: Python 3.10+, ~4 GB free disk space
(torch + transformers are large packages)

1. Install dependencies (one-time):
   pip install -r requirements.txt

2. Run the server:
   python app.py

3. Open your browser:
   http://localhost:5000

===================================================
  Disk space note
===================================================
The full installation (~4 GB) includes:
  • PyTorch (~2 GB)
  • Transformers / DistilBERT (~500 MB)
  • scikit-learn, scipy, lime, flask, etc.

If you get "No space left on device" during pip install,
free up disk space first, then re-run:
  pip install -r requirements.txt

===================================================
  Pages & How to use
===================================================

HERO PAGE
---------
Landing page with animated blob. Click "Start Explaining"
or any feature pill to navigate to an explanation page.

TEXT EXPLANATION PAGE
---------------------
• Sentiment (default) — DistilBERT fine-tuned on SST-2
  Real pre-trained transformer. Downloads ~250 MB model
  weights from HuggingFace on first run, then cached.

• Newsgroups — Logistic Regression trained on the real
  20 Newsgroups dataset (downloads on first run).
  Classifies into: hockey / medicine / religion / graphics

• GPT-4o-mini — paste your OpenAI API key, describe any
  task (e.g. "spam detection") and name the classes.
  LIME sends ~100-150 perturbed texts concurrently.
  Reduce the "LIME Samples" slider to 50 for speed.

TABULAR EXPLANATION PAGE
------------------------
• Iris         — 4-feature flower classifier (Random Forest)
• Wine         — 13-feature wine classifier (Gradient Boosting)
• Breast Cancer — 10-feature tumour classifier (Random Forest)
Adjust the feature value sliders or click "Randomize", then Explain.

CNN IMAGE EXPLANATION PAGE
--------------------------
• ResNet-50, MobileNetV2, VGG-16 (all ImageNet pre-trained)
Upload any JPG/PNG/WEBP photo. LIME masks superpixels
~200 times and ranks regions by impact.
Returns a 4-panel chart:
  1. Original image
  2. Green = regions that support the prediction
  3. Red   = regions that work against the prediction
  4. Heatmap overlay (warm = supports, cool = hurts)

===================================================
  Page navigation
===================================================
All pages transition with smooth fluid animations.
Use the top nav bar (visible on app pages) to switch
between Text / Tabular / Image, or click Home to
return to the Hero page.

===================================================
  OpenAI note
===================================================
When using GPT-4o-mini, each explanation makes ~100-150
parallel API calls (8 threads). This takes 5-15 seconds
and costs roughly 100-150 requests. Reduce the
"LIME Samples" slider to 50 to lower cost and time.
