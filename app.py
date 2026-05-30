import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ARTIFACTS = {
    "model": "cnn_airbnb_model.h5",
    "scaler": "scaler.pkl",
    "label_encoder": "label_encoder_cnn.pkl",
}

DATA_PATH = "new_york_listings_2024.csv"

NUMERIC_FEATURES = [
    "price",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "availability_365",
    "calculated_host_listings_count",
    "latitude",
    "longitude",
    "number_of_reviews_ltm",
    "rating",
    "bedrooms",
    "beds",
    "baths",
]

NEIGHBOURHOOD_GROUPS = [
    "Bronx",
    "Brooklyn",
    "Manhattan",
    "Queens",
    "Staten Island",
]

TOP_NEIGHBOURHOODS = [
    "Bedford-Stuyvesant",
    "Harlem",
    "Williamsburg",
    "Midtown",
    "Hell's Kitchen",
    "Upper East Side",
    "Bushwick",
    "Crown Heights",
    "Upper West Side",
    "East Village",
    "East Flatbush",
    "East Harlem",
    "Chelsea",
    "Astoria",
    "Flushing",
    "Lower East Side",
    "Washington Heights",
    "Greenpoint",
    "Flatbush",
    "East New York",
]

MODEL_FEATURES = (
    NUMERIC_FEATURES
    + [f"neighbourhood_group_{group}" for group in NEIGHBOURHOOD_GROUPS]
    + [f"neighbourhood_{neighbourhood}" for neighbourhood in TOP_NEIGHBOURHOODS]
)

DEFAULTS = {
    "price": 120.0,
    "minimum_nights": 30,
    "number_of_reviews": 10,
    "reviews_per_month": 0.5,
    "availability_365": 180,
    "calculated_host_listings_count": 1,
    "latitude": 40.7306,
    "longitude": -73.9352,
    "number_of_reviews_ltm": 2,
    "rating": 4.7,
    "bedrooms": 1,
    "beds": 1,
    "baths": 1.0,
}


def inject_style():
    st.markdown(
        """
        <style>
        :root {
            --ink: #17202a;
            --muted: #667085;
            --line: #d8dee8;
            --panel: #ffffff;
            --soft: #f6f8fb;
            --brand: #0f766e;
            --brand-dark: #115e59;
            --accent: #f59e0b;
        }

        .stApp {
            background: linear-gradient(180deg, #eef8f6 0%, #f7fafc 42%, #ffffff 100%);
            color: var(--ink);
        }

        .block-container {
            max-width: 920px;
            padding-top: 4.25rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            letter-spacing: 0;
        }

        h1 {
            font-size: 2.35rem !important;
            line-height: 1.12 !important;
            margin-bottom: .35rem !important;
        }

        .app-kicker {
            display: inline-flex;
            align-items: center;
            padding: .32rem .62rem;
            border: 1px solid rgba(15, 118, 110, .22);
            background: rgba(15, 118, 110, .08);
            color: var(--brand-dark);
            border-radius: 999px;
            font-size: .82rem;
            font-weight: 700;
            margin-bottom: .9rem;
        }

        .app-subtitle {
            color: var(--muted);
            max-width: 720px;
            font-size: 1.02rem;
            margin: 0 0 1.35rem 0;
        }

        div[data-testid="stForm"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1.35rem 1.35rem 1.45rem;
            box-shadow: 0 14px 34px rgba(16, 24, 40, .08);
        }

        div[data-testid="stForm"] h3 {
            font-size: 1.15rem !important;
            margin-bottom: .75rem !important;
        }

        div[data-testid="stForm"] label,
        div[data-testid="stForm"] label p {
            color: var(--ink) !important;
            font-weight: 700 !important;
            opacity: 1 !important;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="input"] input {
            background-color: #ffffff !important;
            color: var(--ink) !important;
            border-color: #cfd8e3 !important;
        }

        div[data-baseweb="select"] svg,
        div[data-baseweb="input"] svg {
            color: var(--ink) !important;
            fill: var(--ink) !important;
        }

        div[data-testid="stFormSubmitButton"] button {
            border-radius: 8px;
            border: 0;
            background: var(--brand);
            color: white;
            font-weight: 700;
            min-height: 2.8rem;
            padding-left: 1.35rem;
            padding-right: 1.35rem;
            margin-top: .3rem;
        }

        div[data-testid="stFormSubmitButton"] button:hover {
            background: var(--brand-dark);
            color: white;
        }

        .result-panel {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1.25rem;
            box-shadow: 0 14px 34px rgba(16, 24, 40, .08);
            margin-top: 1.2rem;
        }

        .result-label {
            color: var(--muted);
            font-size: .86rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .04em;
        }

        .result-value {
            color: var(--ink);
            font-size: 2rem;
            line-height: 1.12;
            font-weight: 800;
            margin-top: .2rem;
        }

        .confidence-pill {
            display: inline-flex;
            align-items: center;
            padding: .4rem .7rem;
            background: rgba(245, 158, 11, .14);
            color: #92400e;
            border: 1px solid rgba(245, 158, 11, .28);
            border-radius: 999px;
            font-weight: 800;
            margin-top: .75rem;
        }

        .prob-row {
            margin-top: .85rem;
        }

        .prob-head {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            color: var(--ink);
            font-size: .94rem;
            font-weight: 700;
            margin-bottom: .35rem;
        }

        .prob-track {
            height: .68rem;
            background: #e8edf3;
            border-radius: 999px;
            overflow: hidden;
        }

        .prob-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--brand), var(--accent));
            border-radius: inherit;
        }

        .stDataFrame {
            margin-top: .6rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def enable_numpy_pickle_compatibility():
    """Allow numpy-2 pickles to load on older numpy builds used by TensorFlow."""
    try:
        import numpy.core as numpy_core

        sys.modules.setdefault("numpy._core", numpy_core)
        for module_name in [
            "multiarray",
            "numeric",
            "fromnumeric",
            "shape_base",
            "_multiarray_umath",
        ]:
            try:
                module = __import__(f"numpy.core.{module_name}", fromlist=["*"])
                sys.modules.setdefault(f"numpy._core.{module_name}", module)
            except Exception:
                pass
    except Exception:
        pass


def load_keras_model(path):
    try:
        from tensorflow.keras.models import load_model as keras_load_model
    except Exception:
        from keras.models import load_model as keras_load_model

    return keras_load_model(path, compile=False)


@st.cache_resource
def load_artifacts():
    enable_numpy_pickle_compatibility()
    missing = [path for path in ARTIFACTS.values() if not os.path.exists(path)]
    if missing:
        return None, missing

    artifacts = {
        "model": load_keras_model(ARTIFACTS["model"]),
        "scaler": joblib.load(ARTIFACTS["scaler"]),
        "label_encoder": joblib.load(ARTIFACTS["label_encoder"]),
    }
    return artifacts, []


@st.cache_data
def load_listing_options():
    if not os.path.exists(DATA_PATH):
        return NEIGHBOURHOOD_GROUPS, TOP_NEIGHBOURHOODS

    df = pd.read_csv(DATA_PATH, usecols=["neighbourhood_group", "neighbourhood"])
    groups = sorted(df["neighbourhood_group"].dropna().unique().tolist())
    neighbourhoods = sorted(df["neighbourhood"].dropna().unique().tolist())
    return groups or NEIGHBOURHOOD_GROUPS, neighbourhoods or TOP_NEIGHBOURHOODS


def build_model_input(listing, scaler):
    row = {feature: 0.0 for feature in MODEL_FEATURES}

    for feature in NUMERIC_FEATURES:
        row[feature] = float(listing[feature])

    group_col = f"neighbourhood_group_{listing['neighbourhood_group']}"
    if group_col in row:
        row[group_col] = 1.0

    neighbourhood_col = f"neighbourhood_{listing['neighbourhood']}"
    if neighbourhood_col in row:
        row[neighbourhood_col] = 1.0

    input_df = pd.DataFrame([row], columns=MODEL_FEATURES)

    scale_columns = list(getattr(scaler, "feature_names_in_", []))
    if scale_columns:
        input_df[scale_columns] = scaler.transform(input_df[scale_columns])

    values = input_df.to_numpy(dtype=np.float32)
    return values.reshape(values.shape[0], values.shape[1], 1)


def classify_listing(listing, artifacts):
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    label_encoder = artifacts["label_encoder"]

    x_input = build_model_input(listing, scaler)
    expected_features = model.input_shape[1]
    if x_input.shape[1] != expected_features:
        raise ValueError(
            f"Jumlah fitur input ({x_input.shape[1]}) tidak sama dengan input model ({expected_features})."
        )

    probabilities = model.predict(x_input, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]

    return {
        "label": predicted_label,
        "confidence": float(probabilities[predicted_index]) * 100,
        "probabilities": {
            label: float(probabilities[index]) * 100
            for index, label in enumerate(label_encoder.classes_)
        },
    }


def render_input_form(groups, neighbourhoods):
    with st.form("listing_form"):
        st.subheader("Input Data Listing")

        col1, col2 = st.columns(2)
        with col1:
            neighbourhood_group = st.selectbox(
                "Neighbourhood Group",
                groups,
                index=groups.index("Manhattan") if "Manhattan" in groups else 0,
            )
            neighbourhood = st.selectbox(
                "Neighbourhood",
                neighbourhoods,
                index=neighbourhoods.index("Chelsea") if "Chelsea" in neighbourhoods else 0,
            )
            price = st.number_input("Price", min_value=0.0, value=DEFAULTS["price"], step=10.0)

        with col2:
            minimum_nights = st.number_input(
                "Minimum Nights",
                min_value=1,
                value=DEFAULTS["minimum_nights"],
                step=1,
            )
            rating = st.number_input(
                "Rating",
                min_value=0.0,
                max_value=5.0,
                value=DEFAULTS["rating"],
                step=0.1,
            )
            beds = st.number_input("Beds", min_value=0, value=DEFAULTS["beds"], step=1)

        submitted = st.form_submit_button("Klasifikasikan")

    listing = {
        "neighbourhood_group": neighbourhood_group,
        "neighbourhood": neighbourhood,
        "price": price,
        "minimum_nights": minimum_nights,
        "number_of_reviews": DEFAULTS["number_of_reviews"],
        "reviews_per_month": DEFAULTS["reviews_per_month"],
        "availability_365": DEFAULTS["availability_365"],
        "calculated_host_listings_count": DEFAULTS["calculated_host_listings_count"],
        "latitude": DEFAULTS["latitude"],
        "longitude": DEFAULTS["longitude"],
        "number_of_reviews_ltm": DEFAULTS["number_of_reviews_ltm"],
        "rating": rating,
        "bedrooms": DEFAULTS["bedrooms"],
        "beds": beds,
        "baths": DEFAULTS["baths"],
    }
    return submitted, listing


def render_result(result):
    probabilities = pd.DataFrame(
        {
            "Tipe Akomodasi": list(result["probabilities"].keys()),
            "Probabilitas (%)": list(result["probabilities"].values()),
        }
    ).sort_values("Probabilitas (%)", ascending=False)

    st.markdown('<div class="result-panel">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="result-label">Hasil Klasifikasi</div>
        <div class="result-value">{result["label"]}</div>
        <div class="confidence-pill">Confidence {result["confidence"]:.2f}%</div>
        """,
        unsafe_allow_html=True,
    )

    for _, row in probabilities.iterrows():
        value = float(row["Probabilitas (%)"])
        st.markdown(
            f"""
            <div class="prob-row">
                <div class="prob-head">
                    <span>{row["Tipe Akomodasi"]}</span>
                    <span>{value:.2f}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill" style="width: {max(0, min(value, 100)):.2f}%"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.dataframe(probabilities, hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Klasifikasi Tipe Akomodasi",
        page_icon=":house:",
        layout="centered",
    )
    inject_style()
    st.markdown('<div class="app-kicker">CNN Classification</div>', unsafe_allow_html=True)
    st.title("Klasifikasi Tipe Akomodasi Airbnb NYC")
    st.markdown(
        '<p class="app-subtitle">Masukkan data listing utama untuk memprediksi tipe akomodasi berdasarkan model CNN.</p>',
        unsafe_allow_html=True,
    )

    artifacts, missing = load_artifacts()
    if missing:
        st.error(f"File model belum tersedia: {', '.join(missing)}")
        st.stop()

    groups, neighbourhoods = load_listing_options()
    submitted, listing = render_input_form(groups, neighbourhoods)

    if submitted:
        try:
            with st.spinner("Mengklasifikasikan listing..."):
                result = classify_listing(listing, artifacts)
        except Exception as error:
            st.error(f"Gagal melakukan klasifikasi: {error}")
            st.stop()

        render_result(result)


if __name__ == "__main__":
    main()
