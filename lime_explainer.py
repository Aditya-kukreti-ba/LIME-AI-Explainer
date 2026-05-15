"""
lime_explainer.py - Core LIME explanation engine.
Supports: Text, Tabular, Image explanations.
"""

from __future__ import annotations
import base64, io, traceback
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from lime.lime_text    import LimeTextExplainer
from lime.lime_tabular import LimeTabularExplainer
from lime.lime_image   import LimeImageExplainer

from cnn_models     import predict_single, make_lime_predict_fn, get_imagenet_labels
from models         import get_text_model, get_tabular_model
from openai_wrapper import create_openai_classifier

DARK_BG    = "#0f0f1a"
PANEL_BG   = "#1a1a2e"
BLUE       = "#4cc9f0"
RED        = "#f72585"
GRID_COLOR = "#2a2a4a"
TEXT_COLOR = "#e0e0ff"


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140, facecolor=DARK_BG)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _build_bar_chart(feature_weights, class_names, predicted_class, predicted_prob, title, all_class_probs):
    fw = sorted(feature_weights, key=lambda x: abs(x[1]), reverse=True)[:15]
    if not fw:
        fw = [("(no features)", 0.0)]
    labels  = [f for f, _ in fw]
    weights = [w for _, w in fw]
    colors  = [BLUE if w >= 0 else RED for w in weights]
    n_bars  = len(labels)
    fig_h   = max(5, n_bars * 0.55 + 3.5)
    fig, axes = plt.subplots(1, 2, figsize=(16, fig_h),
                             gridspec_kw={"width_ratios": [3, 1]}, facecolor=DARK_BG)

    ax = axes[0]
    ax.set_facecolor(PANEL_BG)
    bars = ax.barh(range(n_bars), weights, color=colors, alpha=0.85, edgecolor="none", height=0.65)
    ax.set_yticks(range(n_bars))
    ax.set_yticklabels(labels, fontsize=10, color=TEXT_COLOR)
    ax.set_xlabel("Feature weight (impact on prediction)", fontsize=11, color=TEXT_COLOR, labelpad=8)
    ax.set_title(f"{title}\nPrediction: {predicted_class}   confidence: {predicted_prob:.1%}",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=12)
    ax.axvline(0, color=GRID_COLOR, linewidth=1.2, zorder=0)
    ax.spines[:].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.5)
    max_abs = max(abs(w) for w in weights) if weights else 1
    for i, (bar, w) in enumerate(zip(bars, weights)):
        offset = max_abs * 0.025
        ha = "left" if w >= 0 else "right"
        ax.text(w + offset if w >= 0 else w - offset, i, f"{w:+.4f}",
                va="center", ha=ha, fontsize=8, color=TEXT_COLOR)
    pos_patch = mpatches.Patch(color=BLUE, label=f'Supports "{predicted_class}"')
    neg_patch = mpatches.Patch(color=RED,  label=f'Against  "{predicted_class}"')
    ax.legend(handles=[pos_patch, neg_patch], loc="lower right",
              facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9)

    ax2 = axes[1]
    ax2.set_facecolor(PANEL_BG)
    cls_labels = list(all_class_probs.keys())
    cls_probs  = [all_class_probs[c] for c in cls_labels]
    cls_colors = [BLUE if c == predicted_class else "#555577" for c in cls_labels]
    ax2.barh(range(len(cls_labels)), cls_probs, color=cls_colors, alpha=0.85, edgecolor="none", height=0.55)
    ax2.set_yticks(range(len(cls_labels)))
    ax2.set_yticklabels(cls_labels, fontsize=10, color=TEXT_COLOR)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Probability", fontsize=11, color=TEXT_COLOR, labelpad=8)
    ax2.set_title("Class probabilities", fontsize=12, color=TEXT_COLOR, pad=12)
    ax2.spines[:].set_color(GRID_COLOR)
    ax2.tick_params(colors=TEXT_COLOR)
    ax2.grid(axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.5)
    for i, p in enumerate(cls_probs):
        ax2.text(p + 0.02, i, f"{p:.1%}", va="center", fontsize=9, color=TEXT_COLOR)

    plt.tight_layout(pad=1.5)
    return fig


def explain_text(text, model_type, openai_key="", task="sentiment analysis",
                 class_names=None, n_samples=150):
    if class_names is None:
        class_names = ["negative", "positive"]
    try:
        if model_type == "openai":
            if not openai_key:
                return {"error": "OpenAI API key is required for the OpenAI model."}
            predict_fn = create_openai_classifier(openai_key, task, class_names)
        else:
            predict_fn, class_names = get_text_model(model_type)

        explainer = LimeTextExplainer(class_names=class_names, random_state=42)
        exp = explainer.explain_instance(text, predict_fn, num_features=15,
                                         num_samples=n_samples,
                                         labels=list(range(len(class_names))))

        probs         = predict_fn([text])[0]
        predicted_idx = int(np.argmax(probs))
        predicted_cls = class_names[predicted_idx]
        predicted_pb  = float(probs[predicted_idx])
        class_probs   = {c: float(p) for c, p in zip(class_names, probs)}

        try:
            fw = [(w, float(v)) for w, v in exp.as_list(label=predicted_idx)]
        except Exception:
            fw = [(w, float(v)) for w, v in exp.as_list()]

        all_exp = {}
        for i, cls in enumerate(class_names):
            try:
                all_exp[cls] = [(w, float(v)) for w, v in exp.as_list(label=i)]
            except Exception:
                all_exp[cls] = []

        fig     = _build_bar_chart(fw, class_names, predicted_cls, predicted_pb,
                                   title="LIME - Text Explanation",
                                   all_class_probs=class_probs)
        img_b64 = _fig_to_b64(fig)
        return {"success": True, "image": img_b64, "predicted_class": predicted_cls,
                "predicted_prob": predicted_pb, "class_probabilities": class_probs,
                "explanations": all_exp, "model_type": model_type, "task": task}
    except Exception as exc:
        return {"error": str(exc), "traceback": traceback.format_exc()}


def explain_tabular(features, model_type, n_samples=500):
    try:
        model, feature_names, class_names, training_data = get_tabular_model(model_type)
        instance = np.array([features.get(f, 0.0) for f in feature_names], dtype=float)

        explainer = LimeTabularExplainer(training_data, feature_names=feature_names,
                                         class_names=class_names, mode="classification",
                                         discretize_continuous=True, random_state=42)
        exp = explainer.explain_instance(instance, model.predict_proba,
                                         num_features=len(feature_names), num_samples=n_samples)

        probs         = model.predict_proba([instance])[0]
        predicted_idx = int(np.argmax(probs))
        predicted_cls = class_names[predicted_idx]
        predicted_pb  = float(probs[predicted_idx])
        class_probs   = {c: float(p) for c, p in zip(class_names, probs)}
        fw            = [(f, float(w)) for f, w in exp.as_list()]

        fig     = _build_bar_chart(fw, class_names, predicted_cls, predicted_pb,
                                   title=f"LIME - Tabular Explanation ({model_type})",
                                   all_class_probs=class_probs)
        img_b64 = _fig_to_b64(fig)
        return {"success": True, "image": img_b64, "predicted_class": predicted_cls,
                "predicted_prob": predicted_pb, "class_probabilities": class_probs,
                "explanations": fw, "feature_names": feature_names,
                "class_names": class_names, "model_type": model_type}
    except Exception as exc:
        return {"error": str(exc), "traceback": traceback.format_exc()}


def explain_image(image_bytes, model_type="resnet50", n_samples=200, num_features=6):
    """
    LIME image explanation using a pre-trained CNN.
    Returns 4-panel chart: original | green positive | red negative | heatmap
    """
    try:
        import io as _io
        from PIL import Image
        from skimage.segmentation import mark_boundaries

        pil_img   = Image.open(_io.BytesIO(image_bytes)).convert("RGB")
        pil_224   = pil_img.resize((224, 224), Image.LANCZOS)
        img_array = np.array(pil_224)

        top_class, top_prob, top5 = predict_single(model_type, img_array)
        labels  = get_imagenet_labels()
        top_idx = labels.index(top_class) if top_class in labels else 0

        predict_fn = make_lime_predict_fn(model_type)
        explainer  = LimeImageExplainer(random_state=42)
        exp = explainer.explain_instance(img_array, predict_fn, top_labels=5,
                                         hide_color=0, num_samples=n_samples, batch_size=16)

        fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), facecolor=DARK_BG)
        fig.suptitle(
            f'LIME - Image Explanation  |  Model: {model_type}\nPredicted: "{top_class}"   confidence: {top_prob:.1%}',
            fontsize=13, fontweight="bold", color=TEXT_COLOR, y=1.03,
        )
        panel_titles = [
            "Original Image",
            "Supports Prediction\n(green = important regions)",
            "Against Prediction\n(red = misleading regions)",
            "Importance Heatmap\n(warm=supports, cool=hurts)",
        ]
        for ax, t in zip(axes, panel_titles):
            ax.set_facecolor(PANEL_BG)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(t, fontsize=10, color=TEXT_COLOR, pad=6)
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID_COLOR)

        axes[0].imshow(img_array)

        try:
            temp_pos, mask_pos = exp.get_image_and_mask(top_idx, positive_only=True,
                                                        num_features=num_features, hide_rest=True)
            axes[1].imshow(mark_boundaries(temp_pos / 255.0, mask_pos, color=(0.1, 1.0, 0.4), mode="thick"))
        except Exception:
            axes[1].imshow(img_array)

        try:
            temp_neg, mask_neg = exp.get_image_and_mask(top_idx, positive_only=False,
                                                        negative_only=True, num_features=num_features, hide_rest=True)
            axes[2].imshow(mark_boundaries(temp_neg / 255.0, mask_neg, color=(1.0, 0.15, 0.15), mode="thick"))
        except Exception:
            axes[2].imshow(img_array)

        ind_weights = exp.local_exp.get(top_idx, [])
        seg_map     = exp.segments
        heat_map    = np.zeros(seg_map.shape, dtype=float)
        if ind_weights:
            max_abs_w = max(abs(w) for _, w in ind_weights) or 1.0
            for seg_id, w in ind_weights:
                heat_map[seg_map == seg_id] = w / max_abs_w
        axes[3].imshow(img_array, alpha=0.45)
        hm   = axes[3].imshow(heat_map, cmap="RdYlGn", alpha=0.72, vmin=-1, vmax=1)
        cbar = fig.colorbar(hm, ax=axes[3], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=7)
        cbar.outline.set_edgecolor(GRID_COLOR)

        plt.tight_layout()
        img_b64 = _fig_to_b64(fig)

        sorted_weights = sorted(ind_weights, key=lambda x: abs(x[1]), reverse=True)[:10]
        sw_list = [{"segment": int(sid), "weight": float(w)} for sid, w in sorted_weights]

        return {"success": True, "image": img_b64, "predicted_class": top_class,
                "predicted_prob": float(top_prob),
                "top5": [{"class": c, "prob": float(p)} for c, p in top5],
                "segment_weights": sw_list, "model_type": model_type}
    except Exception as exc:
        return {"error": str(exc), "traceback": traceback.format_exc()}
