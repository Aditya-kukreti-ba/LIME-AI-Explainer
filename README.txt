===================================================
  LIME Explainer — Quick Start
===================================================

1. Install dependencies (one-time):
   pip install -r requirements.txt

2. Run the server:
   python app.py

3. Open your browser:
   http://localhost:5000

===================================================
  How to use
===================================================

TEXT TAB
--------
• Sentiment model   — built-in TF-IDF + Logistic Regression
• Newsgroups model  — 20-category topic classifier (downloads dataset on first run)
• GPT-4o-mini       — paste your OpenAI API key, describe any task
                      (e.g. "spam detection", "toxicity classification")
                      and name the classes ("spam, ham" or "toxic, safe")

TABULAR TAB
-----------
• Iris       — 4-feature flower classifier (Random Forest)
• Wine       — 13-feature wine-variety classifier (Gradient Boosting)
• Breast Cancer — 10-feature tumour classifier (Random Forest)
Adjust sliders or click "Randomize values", then hit Explain.

===================================================
  What you get
===================================================
• A real matplotlib LIME bar chart (positive = supports prediction,
  negative = works against it)
• Class probability bars for all classes
• A feature-weight table with mini bar-charts
• Everything runs locally — no dummy data

===================================================
  OpenAI note
===================================================
When using GPT-4o-mini, LIME sends ~100-150 perturbed text
variants to the API concurrently (8 threads).  This typically
takes 5-15 seconds and uses roughly 100-150 API calls per
explanation.  Reduce "LIME Samples" slider to 50 for speed.
