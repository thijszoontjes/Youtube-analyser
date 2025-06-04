import os
import sys
import argparse
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from joblib import dump, load

import streamlit as st
import shap

# ───── Configuration ─────────────────────────────────────────────────────────

RAW_PATH = Path("data/raw")
MODEL_PATH = Path("models/model.pkl")
FEATURES = [
    "title_len",
    "emoji_cnt",
    "upload_hour",
    "impressions",
    "ctr",
    "subs_at_upload",
]

# ───── Data Loading & Feature Engineering ────────────────────────────────────

def load_csvs(path: Path) -> pd.DataFrame:
    """
    Read all CSV files in `data/raw/` and concatenate into a single DataFrame.
    Raises FileNotFoundError if no CSV is present.
    """
    files = list(path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV exports found in {path}")
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Compute title length and emoji count from 'Video title'.
    - Fill upload_hour with a placeholder of 17 (5 PM).
    - Rename the relevant columns to match FEATURES:
       • Views → views_7d
       • Impressions → impressions
       • Impressions click-through rate (%) → ctr
       • Subscribers → subs_at_upload
    """
    df = df.copy()
    # Compute title length and emoji count
    df["title_len"] = df["Video title"].str.len()
    df["emoji_cnt"] = df["Video title"].str.count(r"[^\w\s,]")

    # Use a fixed upload hour (placeholder)
    df["upload_hour"] = 17

    # Rename columns so they match our model’s expected feature names
    df.rename(
        columns={
            "Views": "views_7d",
            "Impressions": "impressions",
            "Impressions click-through rate (%)": "ctr",
            "Subscribers": "subs_at_upload",
        },
        inplace=True,
    )

    return df

# ───── Model Training & Saving ───────────────────────────────────────────────

def train_and_save() -> float:
    """
    1. Load raw CSV data from data/raw/
    2. Engineer features
    3. Split into train/test
    4. Fit a LightGBM regressor
    5. Print and return R² on the test set
    6. Save the model (and feature list) to models/model.pkl
    """
    df = feature_engineer(load_csvs(RAW_PATH))
    X, y = df[FEATURES], df["views_7d"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LGBMRegressor(n_estimators=400, learning_rate=0.07)
    model.fit(X_train, y_train)

    r2 = model.score(X_test, y_test)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump({"model": model, "features": FEATURES}, MODEL_PATH)

    return r2

def cli_train():
    """
    Command-line entrypoint: trains the model and prints R², or
    prints an error if no CSV was found.
    """
    try:
        r2 = train_and_save()
        print(f"Model trained – R²: {r2:.3f}")
    except FileNotFoundError as e:
        print(f"⚠️  {e}")

# ───── Helper: Simple Explanation ────────────────────────────────────────────

def explain_simple(model, X_row: pd.DataFrame) -> str:
    """
    Generate a grandmother-friendly explanation of the prediction.
    - States the base guess (if no features).
    - Describes how each detail (title length, CTR, etc.) raised or lowered the guess,
      using simple sentences.
    - Ends with very simple advice (“shorten your title,” “keep using emojis,” etc.).
    """
    # Compute raw prediction and SHAP values
    pred_value = model.predict(X_row)[0]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_row)[0]
    base_value = explainer.expected_value

    # Gather feature contributions
    contributions = list(zip(X_row.columns, shap_values))

    # Build plain English sentences
    lines = []
    # Explain base guess
    lines.append(
        f"If I knew nothing about your video, I'd guess around {base_value:.0f} views "
        "in the first week."
    )

    # Process each feature's effect:
    # title_len, emoji_cnt, upload_hour, impressions, ctr, subs_at_upload
    tl = int(X_row["title_len"].iloc[0])
    ec = int(X_row["emoji_cnt"].iloc[0])
    uh = int(X_row["upload_hour"].iloc[0])
    imp = int(X_row["impressions"].iloc[0])
    ctr_val = float(X_row["ctr"].iloc[0])
    subs = int(X_row["subs_at_upload"].iloc[0])

    # Title length effect
    # Find SHAP value for title_len
    title_shap = next(val for feat, val in contributions if feat == "title_len")
    if title_shap < 0:
        lines.append(
            f"Because your title is {tl} characters long, I lower my guess by "
            f"{abs(title_shap):.0f} views. Longer titles often get fewer clicks."
        )
    else:
        lines.append(
            f"Since your title is {tl} characters long, I add {title_shap:.0f} views. "
            "That length seems nice."
        )

    # CTR effect
    ctr_shap = next(val for feat, val in contributions if feat == "ctr")
    if ctr_shap < 0:
        lines.append(
            f"If fewer people click your thumbnail (CTR {ctr_val:.1f}%), I take away "
            f"{abs(ctr_shap):.0f} views."
        )
    else:
        lines.append(
            f"Because your CTR is {ctr_val:.1f}%, I add {ctr_shap:.0f} views. "
            "People clicking a lot is a good sign."
        )

    # Impressions effect
    imp_shap = next(val for feat, val in contributions if feat == "impressions")
    if imp_shap < 0:
        lines.append(
            f"With {imp} impressions, I lower my guess by {abs(imp_shap):.0f} views. "
            "Showing too many times without clicks can hurt."
        )
    else:
        lines.append(
            f"Since you have {imp} impressions, I add {imp_shap:.0f} views. "
            "That’s good exposure."
        )

    # Emoji count effect
    ec_shap = next(val for feat, val in contributions if feat == "emoji_cnt")
    if ec_shap < 0:
        lines.append(
            f"With {ec} emojis in your title, I surprisingly lower my guess by "
            f"{abs(ec_shap):.0f}. Maybe too many emojis can confuse people."
        )
    else:
        lines.append(
            f"Since you used {ec} emoji{'s' if ec != 1 else ''}, I add {ec_shap:.0f} views. "
            "A little color in the title can help."
        )

    # Subscribers effect
    subs_shap = next(val for feat, val in contributions if feat == "subs_at_upload")
    if subs_shap < 0:
        lines.append(
            f"You had {subs} subscribers when you uploaded, so I lower my guess by "
            f"{abs(subs_shap):.0f} views. More subscribers usually helps more."
        )
    else:
        lines.append(
            f"With {subs} subscribers, I add {subs_shap:.0f} views. "
            "That many subscribers is a good audience."
        )

    # Upload hour effect
    uh_shap = next(val for feat, val in contributions if feat == "upload_hour")
    if uh_shap < 0:
        lines.append(
            f"Uploading at hour {uh}, I lower my guess by {abs(uh_shap):.0f} views. "
            "That hour might be a bit off-peak."
        )
    else:
        lines.append(
            f"Since you upload at hour {uh}, I add {uh_shap:.0f} views. "
            "That seems like a neutral or good time."
        )

    # Summarize final prediction
    lines.append("")
    lines.append(f"All together, I now guess about {pred_value:.0f} views in the first week.")

    # Simple actionable tips
    lines.append("")
    lines.append("Here are some simple tips you can follow:")
    # Tip: shorten or lengthen title
    if tl < 20:
        lines.append(" • Your title is quite short. Try making it a bit longer so people know what it’s about.")
    elif tl > 60:
        lines.append(" • Your title is very long. Try shortening it so it’s easier to read.")
    else:
        lines.append(" • Your title length is okay.")

    # Tip: emojis
    if ec == 0:
        lines.append(" • You didn’t use any emoji. Adding one could catch people’s eye.")
    elif ec > 3:
        lines.append(" • You used a lot of emojis. Maybe limit to one or two to keep it clear.")
    else:
        lines.append(" • Your emoji usage is fine.")

    # Tip: upload hour
    if uh not in (17, 18, 19):
        lines.append(" • Consider uploading between 5 PM and 7 PM, when more people are watching.")
    else:
        lines.append(" • Uploading between 5 PM and 7 PM is good timing.")

    # Tip: CTR
    if ctr_val < 4.0:
        lines.append(" • Your click rate is under 4 %. Try a new thumbnail to get more clicks.")
    else:
        lines.append(" • Your click rate is good. Keep your thumbnail style.")

    # Tip: subscribers
    if subs < 1000:
        lines.append(" • With under 1000 subscribers, keep promoting your channel gently to grow.")
    else:
        lines.append(" • Your subscriber count is healthy; keep engaging your audience.")

    return "\n\n".join(lines)

# ───── Streamlit App ─────────────────────────────────────────────────────────

def load_model():
    """
    Load the saved model from models/model.pkl if it exists.
    Returns (model, features) or (None, None) if not found.
    """
    if MODEL_PATH.exists():
        pack = load(MODEL_PATH)
        return pack["model"], pack["features"]
    return None, None

def streamlit_app():
    st.set_page_config(page_title="YouTube Video Analyzer", layout="wide")
    st.title("📊 YouTube Video Analyzer")

    # Sidebar: Train / Retrain the model
    if st.sidebar.button("Train / retrain model"):
        with st.spinner("Training model…"):
            try:
                r2 = train_and_save()
                st.sidebar.success(f"Model trained (R² = {r2:.3f})")
                st.experimental_rerun()
            except FileNotFoundError as e:
                st.sidebar.error(str(e))

    # Load the model (if it exists)
    model, feat_cols = load_model()

    # Input form
    title_input = st.text_input("Video title (so we can count characters and emojis)")
    upload_hour_input = st.slider("Planned upload hour (0 – 23)", 0, 23, 17)
    impressions_input = st.number_input(
        "Expected first-day impressions", min_value=0, step=100, value=1000
    )
    ctr_input = st.slider("Target CTR (%)", 0.0, 20.0, 4.5, 0.1)
    subs_input = st.number_input(
        "Current subscribers", min_value=0, step=100, value=5000
    )

    # Predict button
    if st.button("Predict & Explain"):
        # Ensure the model is trained
        if model is None:
            st.error("No model found. Please train the model first via the sidebar.")
            st.stop()

        # Build a single-row DataFrame with the same columns as FEATURES
        X_new = pd.DataFrame([
            {
                "title_len": len(title_input),
                "emoji_cnt": sum(
                    1 for c in title_input if not c.isalnum() and not c.isspace()
                ),
                "upload_hour": upload_hour_input,
                "impressions": impressions_input,
                "ctr": ctr_input,
                "subs_at_upload": subs_input,
            }
        ])

        # Get the friendly explanation
        explanation_text = explain_simple(model, X_new)

        # Display the text as markdown
        st.markdown(explanation_text)

# ───── Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", action="store_true",
        help="Train or retrain the model and exit"
    )
    args = parser.parse_args()
    if args.train:
        cli_train()
        sys.exit()
    streamlit_app()
