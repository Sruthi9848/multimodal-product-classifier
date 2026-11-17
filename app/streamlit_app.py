"""
Streamlit web app — classify a product from its name and/or photo.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                      # make `src` importable
sys.path.insert(0, str(Path(__file__).parent))     # make `ui_components` importable

import random  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402

import ui_components as ui  # noqa: E402
from src.config import Config  # noqa: E402
from src.utils import load_json  # noqa: E402

st.set_page_config(page_title="Product classifier · LoRA", page_icon="🏷️", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)

CFG = Config()
MODE_LABELS = {"multimodal": "Name + photo (multimodal)", "text": "Name only", "image": "Photo only"}


# ---------------------------------------------------------------------- #
# Cached loaders (run once per session, not on every click)
# ---------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading the model (first time downloads the pretrained backbones)…")
def load_classifier(checkpoint: str, _mtime: float):
    """`_mtime` makes the cache refresh automatically after you retrain."""
    from src.inference.predictor import ProductClassifier
    return ProductClassifier(checkpoint)


@st.cache_resource(show_spinner="Loading test products…")
def load_test_split():
    if not CFG.splits_file.exists():
        return None
    from src.data.dataset_loader import get_split_datasets
    return get_split_datasets(CFG)["test"]


def available_checkpoints() -> dict:
    return {m: CFG.checkpoint_file(m) for m in MODE_LABELS if CFG.checkpoint_file(m).exists()}


# ---------------------------------------------------------------------- #
# Sidebar
# ---------------------------------------------------------------------- #
ckpts = available_checkpoints()
with st.sidebar:
    st.markdown("### Model")
    if ckpts:
        mode = st.radio("Which inputs should the model use?", list(ckpts),
                        format_func=lambda m: MODE_LABELS[m])
        top_k = st.slider("Alternatives to show", 3, 10, 5)
        st.caption(f"Checkpoint: `{ckpts[mode].relative_to(ROOT)}`")
    else:
        mode, top_k = None, 5
    st.markdown("---")
    st.caption("DistilBERT reads the product name, a Vision Transformer reads the photo. "
               "Both were fine-tuned with LoRA, which trains under 2% of their weights.")

st.markdown(ui.hero("What kind of product is this?",
                    "Give a product name, a photo, or both. A LoRA-tuned DistilBERT + ViT model "
                    "predicts its catalogue category and how sure it is."), unsafe_allow_html=True)
st.write("")

tab_classify, tab_perf, tab_how = st.tabs(["Classify a product", "Model performance", "How it works"])

# ====================================================================== #
# TAB 1 — Classify
# ====================================================================== #
with tab_classify:
    if not ckpts:
        st.markdown(ui.empty_state(
            "No trained model yet",
            "Train one from the project root, then refresh this page:<br><br>"
            "<code>python scripts/prepare_data.py</code><br><code>python scripts/train.py</code>"),
            unsafe_allow_html=True)
        st.stop()

    clf = load_classifier(str(ckpts[mode]), ckpts[mode].stat().st_mtime)
    if "example" not in st.session_state:
        st.session_state.example = None

    left, right = st.columns([5, 6], gap="large")

    with left:
        st.markdown("#### Your product")
        if st.button("Try a random test-set product", help="Picks an item the model never saw in training"):
            test = load_test_split()
            if test is None:
                st.warning("Run `python scripts/prepare_data.py` first to create the test split.")
            else:
                row = test[random.randrange(len(test))]
                st.session_state.example = {"text": row[CFG.text_column], "image": row[CFG.image_column],
                                            "truth": row[CFG.label_column]}
                st.session_state.product_name = row[CFG.text_column]

        ex = st.session_state.example
        name = st.text_input("Product name or short description", key="product_name",
                             placeholder="e.g. Titan Women Silver Analog Watch",
                             disabled=(mode == "image"))
        upload = st.file_uploader("Product photo", type=["jpg", "jpeg", "png", "webp"],
                                  disabled=(mode == "text"))

        image = None
        if upload is not None:
            try:
                image = Image.open(upload)
                st.session_state.example = ex = None   # a new upload replaces the example photo
            except UnidentifiedImageError:
                st.error("That file isn't a readable image. Upload a JPG, PNG or WebP photo.")
        elif ex is not None and mode != "text":
            image = ex["image"]

        if image is not None and mode != "text":
            st.image(image, caption="Photo sent to the model", width=220)

        run = st.button("Classify product", type="primary", width="stretch")

    with right:
        st.markdown("#### Prediction")
        text_in = name if mode != "image" else None
        img_in = image if mode != "text" else None

        if run:
            try:
                st.session_state.result = clf.predict(text=text_in, image=img_in, top_k=top_k)
                st.session_state.result_truth = ex["truth"] if ex else None
            except ValueError as err:
                st.session_state.result = None
                st.warning(str(err))

        res = st.session_state.get("result")
        if res:
            st.markdown(ui.prediction_tag(res["label"], res["confidence"], res["inputs_used"]),
                        unsafe_allow_html=True)
            truth = st.session_state.get("result_truth")
            if truth:
                (st.success if truth == res["label"] else st.error)(
                    f"Catalogue label for this test product: **{truth}**")
            if res["confidence"] < 0.5:
                st.info("Low confidence. The model is split between several categories; "
                        "check the alternatives below or add the missing input.")
            st.markdown(ui.reliance_bar(res["text_weight"]), unsafe_allow_html=True)
            st.markdown('<div class="muted" style="margin-top:14px">Top alternatives</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(ui.topk_chart(res["top_k"]), width="stretch",
                            config={"displayModeBar": False})
        else:
            st.markdown(ui.empty_state(
                "Nothing classified yet",
                "Type a product name, add a photo, then press <em>Classify product</em>. "
                "Or try a random product from the test set."), unsafe_allow_html=True)

    with st.expander(f"Categories this model knows ({len(clf.class_names)})"):
        st.write(", ".join(clf.class_names))

# ====================================================================== #
# TAB 2 — Performance (real numbers only, read from outputs/metrics)
# ====================================================================== #
with tab_perf:
    metric_files = sorted(CFG.metrics_dir.glob("lora_*.json")) + sorted(CFG.metrics_dir.glob("baseline_*.json"))
    if not metric_files:
        st.markdown(ui.empty_state(
            "No evaluation results yet",
            "Results appear here after you run:<br><br><code>python scripts/train.py</code><br>"
            "<code>python scripts/train_baseline.py</code><br><code>python scripts/compare_models.py</code>"),
            unsafe_allow_html=True)
    else:
        results = {f.stem: load_json(f) for f in metric_files}
        choice = st.selectbox("Model", list(results), format_func=lambda k: results[k]["model"])
        r = results[choice]
        kpis = [("Accuracy", f"{r['accuracy']:.1%}"), ("Macro F1", f"{r['f1_macro']:.3f}"),
                ("Macro precision", f"{r['precision_macro']:.3f}"), ("Macro recall", f"{r['recall_macro']:.3f}")]
        if r.get("top3_accuracy") is not None:
            kpis.append(("Top-3 accuracy", f"{r['top3_accuracy']:.1%}"))
        st.markdown(ui.kpi_strip(kpis), unsafe_allow_html=True)
        st.caption(f"Held-out test set: {r.get('num_test_samples', '?'):,} products"
                   + (f" · trainable parameters: {r['trainable_params']:,} of {r['total_params']:,} "
                      f"({r['trainable_pct']:.2f}%)" if r.get("trainable_params") else ""))

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("##### F1 by category")
            st.plotly_chart(ui.per_class_f1_chart(r["per_class"], r["class_names"]),
                            width="stretch", config={"displayModeBar": False})
        with c2:
            st.markdown("##### Confusion matrix")
            cm_png = CFG.figures_dir / f"cm_{choice}.png"
            if cm_png.exists():
                st.image(str(cm_png), width="stretch")

        st.markdown("##### All models compared")
        table = pd.DataFrame([{
            "Model": v["model"], "Accuracy": v["accuracy"], "Macro F1": v["f1_macro"],
            "Weighted F1": v["f1_weighted"], "Macro precision": v["precision_macro"],
            "Macro recall": v["recall_macro"], "Top-3 acc.": v.get("top3_accuracy"),
        } for v in results.values()]).sort_values("Macro F1", ascending=False)
        st.dataframe(table.style.format(precision=4, na_rep="—"), hide_index=True, width="stretch")
        hist_png = CFG.figures_dir / f"history_{r.get('run_name', r['mode'])}.png"
        if r["family"] == "lora" and hist_png.exists():
            with st.expander("Training curves"):
                st.image(str(hist_png), width="stretch")

# ====================================================================== #
# TAB 3 — How it works
# ====================================================================== #
with tab_how:
    st.markdown("##### Architecture")
    st.graphviz_chart("""
    digraph G {
      rankdir=LR; bgcolor="transparent"; node [shape=box, style="filled", fontname="Helvetica", fontsize=11,
      color="#1E2B4A", fillcolor="#FFFFFF", fontcolor="#1E2B4A"]; edge [color="#5B6B86"];
      name [label="Product name"]; photo [label="Product photo"];
      bert [label="DistilBERT\\n+ LoRA (q, v)", fillcolor="#EDF0F5"];
      vit  [label="ViT-B/16\\n+ LoRA (q, v)", fillcolor="#EDF0F5"];
      fuse [label="Gated fusion\\nz = g·t + (1−g)·v", fillcolor="#F2B630"];
      head [label="Classifier\\nsoftmax"]; out [label="Category +\\nconfidence"];
      name -> bert -> fuse; photo -> vit -> fuse; fuse -> head -> out;
    }""")
    st.markdown("##### LoRA in one line")
    st.latex(r"h = W_0 x + \frac{\alpha}{r}\,B A x,\qquad A\in\mathbb{R}^{r\times k},\; B\in\mathbb{R}^{d\times r},\; r \ll d,k")
    st.markdown(
        "The pretrained weight $W_0$ stays frozen. Only the two thin matrices $A$ and $B$ are learned, "
        f"with rank $r={CFG.lora_r}$. For a 768×768 attention projection that is "
        f"{CFG.lora_r * (768 + 768):,} trainable numbers instead of {768 * 768:,}.")
    st.markdown("##### Confidence")
    st.latex(r"p(c \mid x) = \frac{e^{z_c}}{\sum_j e^{z_j}}")
    st.markdown("The confidence shown on the tag is the softmax probability of the top category. "
                "The data is the public *Fashion Product Images (Small)* catalogue (≈44k products), "
                "via the Hugging Face dataset `ashraq/fashion-product-images-small`.")
