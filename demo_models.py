"""
demo_models.py
Pre-trained sklearn demo models for LIME explanations.
Models are cached in memory after first load to avoid retraining.
"""

from sklearn.datasets import load_iris, load_wine, load_breast_cancer, fetch_20newsgroups
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# ── In-memory cache ───────────────────────────────────────────────────────────
_model_cache: dict = {}


# ── Tabular models ────────────────────────────────────────────────────────────

TABULAR_MODEL_INFO = [
    {
        "id": "iris",
        "name": "Iris Classifier (Random Forest)",
        "description": "Classifies iris flowers by petal and sepal measurements.",
        "features": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
        "classes": ["setosa", "versicolor", "virginica"],
        "feature_ranges": {
            "sepal length (cm)": {"min": 4.3, "max": 7.9, "default": 5.8, "step": 0.1},
            "sepal width (cm)":  {"min": 2.0, "max": 4.4, "default": 3.0, "step": 0.1},
            "petal length (cm)": {"min": 1.0, "max": 6.9, "default": 3.7, "step": 0.1},
            "petal width (cm)":  {"min": 0.1, "max": 2.5, "default": 1.2, "step": 0.1},
        },
    },
    {
        "id": "wine",
        "name": "Wine Classifier (Gradient Boosting)",
        "description": "Classifies wine varieties by chemical properties.",
        "features": [
            "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
            "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
            "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline",
        ],
        "classes": ["class_0", "class_1", "class_2"],
        "feature_ranges": {
            "alcohol":                       {"min": 11.0,  "max": 15.0,   "default": 13.0,  "step": 0.1},
            "malic_acid":                    {"min": 0.7,   "max": 5.8,    "default": 2.3,   "step": 0.1},
            "ash":                           {"min": 1.4,   "max": 3.2,    "default": 2.3,   "step": 0.05},
            "alcalinity_of_ash":             {"min": 10.6,  "max": 30.0,   "default": 19.5,  "step": 0.5},
            "magnesium":                     {"min": 70.0,  "max": 162.0,  "default": 99.7,  "step": 1.0},
            "total_phenols":                 {"min": 0.98,  "max": 3.88,   "default": 2.3,   "step": 0.05},
            "flavanoids":                    {"min": 0.34,  "max": 5.08,   "default": 2.0,   "step": 0.05},
            "nonflavanoid_phenols":          {"min": 0.13,  "max": 0.66,   "default": 0.36,  "step": 0.01},
            "proanthocyanins":               {"min": 0.41,  "max": 3.58,   "default": 1.59,  "step": 0.05},
            "color_intensity":               {"min": 1.28,  "max": 13.0,   "default": 5.06,  "step": 0.1},
            "hue":                           {"min": 0.48,  "max": 1.71,   "default": 0.96,  "step": 0.01},
            "od280/od315_of_diluted_wines":  {"min": 1.27,  "max": 4.0,    "default": 2.61,  "step": 0.05},
            "proline":                       {"min": 278.0, "max": 1680.0, "default": 746.0, "step": 10.0},
        },
    },
    {
        "id": "breast_cancer",
        "name": "Breast Cancer Classifier (Random Forest)",
        "description": "Classifies tumours as malignant or benign from digitized biopsy features.",
        "features": [
            "mean radius", "mean texture", "mean perimeter", "mean area",
            "mean smoothness", "mean compactness", "mean concavity",
            "mean concave points", "mean symmetry", "mean fractal dimension",
        ],
        "classes": ["malignant", "benign"],
        "feature_ranges": {
            "mean radius":           {"min": 6.9,   "max": 28.1,   "default": 14.1,  "step": 0.1},
            "mean texture":          {"min": 9.7,   "max": 39.3,   "default": 19.3,  "step": 0.1},
            "mean perimeter":        {"min": 43.8,  "max": 188.5,  "default": 91.9,  "step": 0.5},
            "mean area":             {"min": 143.5, "max": 2501.0, "default": 654.9, "step": 5.0},
            "mean smoothness":       {"min": 0.05,  "max": 0.16,   "default": 0.096, "step": 0.001},
            "mean compactness":      {"min": 0.02,  "max": 0.35,   "default": 0.104, "step": 0.005},
            "mean concavity":        {"min": 0.0,   "max": 0.43,   "default": 0.089, "step": 0.005},
            "mean concave points":   {"min": 0.0,   "max": 0.20,   "default": 0.049, "step": 0.002},
            "mean symmetry":         {"min": 0.11,  "max": 0.30,   "default": 0.181, "step": 0.002},
            "mean fractal dimension":{"min": 0.05,  "max": 0.10,   "default": 0.063, "step": 0.001},
        },
    },
]


def _load_tabular(model_id: str):
    key = f"tabular_{model_id}"
    if key in _model_cache:
        return _model_cache[key]

    if model_id == "iris":
        ds = load_iris()
        X, y = ds.data, ds.target
        feature_names = list(ds.feature_names)
        class_names   = list(ds.target_names)
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    elif model_id == "wine":
        ds = load_wine()
        X, y = ds.data, ds.target
        feature_names = [f.replace(" ", "_") for f in ds.feature_names]
        class_names   = list(ds.target_names)
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)

    elif model_id == "breast_cancer":
        ds = load_breast_cancer()
        # Use only the first 10 "mean" features for simplicity in the UI
        X  = ds.data[:, :10]
        y  = ds.target
        feature_names = list(ds.feature_names[:10])
        class_names   = list(ds.target_names)
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    else:
        raise ValueError(f"Unknown tabular model: {model_id!r}")

    model.fit(X, y)
    result = (model, feature_names, class_names, X)
    _model_cache[key] = result
    return result


def get_tabular_model(model_id: str):
    """Return (model, feature_names, class_names, training_data)."""
    return _load_tabular(model_id)


def get_tabular_model_info():
    return TABULAR_MODEL_INFO


def get_tabular_model_info_by_id(model_id: str):
    for info in TABULAR_MODEL_INFO:
        if info["id"] == model_id:
            return info
    raise ValueError(f"Unknown model id: {model_id!r}")


# ── Text models ───────────────────────────────────────────────────────────────

TEXT_MODEL_INFO = [
    {
        "id": "newsgroups",
        "name": "20 Newsgroups Topic Classifier (Logistic Regression)",
        "description": "Classifies text into four topic categories from the 20 Newsgroups dataset.",
        "classes": ["hockey", "medicine", "religion", "graphics"],
        "example": "The NHL game last night was incredible, the goalie made some amazing saves.",
    },
    {
        "id": "sentiment",
        "name": "Sentiment Classifier (Logistic Regression)",
        "description": "Classifies text as positive or negative sentiment using TF-IDF features.",
        "classes": ["negative", "positive"],
        "example": "This product is absolutely fantastic! I love every feature of it.",
    },
]


def _load_text_newsgroups():
    key = "text_newsgroups"
    if key in _model_cache:
        return _model_cache[key]

    categories = [
        "rec.sport.hockey",
        "sci.med",
        "soc.religion.christian",
        "comp.graphics",
    ]
    class_names = ["hockey", "medicine", "religion", "graphics"]

    print("[demo_models] Downloading 20newsgroups (first run only)…")
    train = fetch_20newsgroups(
        subset="train", categories=categories,
        remove=("headers", "footers", "quotes"),
    )
    vectorizer = TfidfVectorizer(max_features=10_000, stop_words="english", ngram_range=(1, 2))
    X = vectorizer.fit_transform(train.data)
    model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    model.fit(X, train.target)

    result = (model, vectorizer, class_names)
    _model_cache[key] = result
    return result


def _load_text_sentiment():
    """Tiny hand-crafted sentiment model trained on synthetic data for demo."""
    key = "text_sentiment"
    if key in _model_cache:
        return _model_cache[key]

    positive_texts = [
        "This is absolutely wonderful and amazing",
        "I love this product it is fantastic and great",
        "Excellent quality, very happy with my purchase",
        "Outstanding performance, highly recommend",
        "Best thing I have ever bought, superb",
        "Brilliant, awesome, incredible experience",
        "Perfect, delightful, beautiful result",
        "Great value, top quality, very satisfied",
        "Loved it, fantastic service, will buy again",
        "Exceptional, remarkable, truly impressive",
        "Wonderful experience, very pleased overall",
        "Superb quality and amazing customer support",
        "Highly impressed with the excellent results",
        "Fantastic product that exceeded expectations",
        "Brilliant design and outstanding performance",
    ]
    negative_texts = [
        "This is terrible and awful, I hate it",
        "Worst product ever, completely useless junk",
        "Disappointed and frustrated, poor quality",
        "Horrible experience, do not recommend at all",
        "Broken on arrival, disgusting customer service",
        "Terrible waste of money, very disappointed",
        "Awful quality, stopped working after a day",
        "Dreadful product, absolutely not worth it",
        "Pathetic, useless, complete waste of time",
        "Miserable failure, nothing worked properly",
        "Very bad quality, extremely dissatisfied",
        "Broken, defective, and impossible to fix",
        "Horrendous experience, regret buying this",
        "Inferior quality with terrible support team",
        "Complete disaster, avoid this product please",
    ]

    texts  = positive_texts + negative_texts
    labels = [1] * len(positive_texts) + [0] * len(negative_texts)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression(max_iter=500, C=1.0, random_state=42)
    model.fit(X, labels)

    class_names = ["negative", "positive"]
    result = (model, vectorizer, class_names)
    _model_cache[key] = result
    return result


def get_text_model(model_id: str):
    """Return (model, vectorizer, class_names)."""
    if model_id == "newsgroups":
        return _load_text_newsgroups()
    elif model_id == "sentiment":
        return _load_text_sentiment()
    else:
        raise ValueError(f"Unknown text model: {model_id!r}")


def get_text_model_info():
    return TEXT_MODEL_INFO
