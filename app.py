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
            minimum_nights = st.number_input(
                "Minimum Nights",
                min_value=1,
                value=DEFAULTS["minimum_nights"],
                step=1,
            )
            availability_365 = st.number_input(
                "Availability 365",
                min_value=0,
                max_value=365,
                value=DEFAULTS["availability_365"],
                step=1,
            )
            latitude = st.number_input(
                "Latitude",
                value=DEFAULTS["latitude"],
                format="%.6f",
            )
            longitude = st.number_input(
                "Longitude",
                value=DEFAULTS["longitude"],
                format="%.6f",
            )

        with col2:
            number_of_reviews = st.number_input(
                "Number of Reviews",
                min_value=0,
                value=DEFAULTS["number_of_reviews"],
                step=1,
            )
            reviews_per_month = st.number_input(
                "Reviews per Month",
                min_value=0.0,
                value=DEFAULTS["reviews_per_month"],
                step=0.1,
            )
            number_of_reviews_ltm = st.number_input(
                "Reviews Last 12 Months",
                min_value=0,
                value=DEFAULTS["number_of_reviews_ltm"],
                step=1,
            )
            calculated_host_listings_count = st.number_input(
                "Host Listings Count",
                min_value=1,
                value=DEFAULTS["calculated_host_listings_count"],
                step=1,
            )
            rating = st.number_input(
                "Rating",
                min_value=0.0,
                max_value=5.0,
                value=DEFAULTS["rating"],
                step=0.1,
            )
            bedrooms = st.number_input("Bedrooms", min_value=0, value=DEFAULTS["bedrooms"], step=1)
            beds = st.number_input("Beds", min_value=0, value=DEFAULTS["beds"], step=1)
            baths = st.number_input("Baths", min_value=0.0, value=DEFAULTS["baths"], step=0.5)

        submitted = st.form_submit_button("Klasifikasikan")

    listing = {
        "neighbourhood_group": neighbourhood_group,
        "neighbourhood": neighbourhood,
        "price": price,
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "reviews_per_month": reviews_per_month,
        "availability_365": availability_365,
        "calculated_host_listings_count": calculated_host_listings_count,
        "latitude": latitude,
        "longitude": longitude,
        "number_of_reviews_ltm": number_of_reviews_ltm,
        "rating": rating,
        "bedrooms": bedrooms,
        "beds": beds,
        "baths": baths,
    }
    return submitted, listing


def main():
    st.set_page_config(page_title="Klasifikasi Tipe Akomodasi", page_icon=":house:")
    st.title("Klasifikasi Tipe Akomodasi Airbnb NYC")

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

        st.subheader("Hasil Klasifikasi")
        st.metric("Tipe Akomodasi", result["label"], f"{result['confidence']:.2f}%")

        probabilities = pd.DataFrame(
            {
                "Tipe Akomodasi": list(result["probabilities"].keys()),
                "Probabilitas (%)": list(result["probabilities"].values()),
            }
        ).sort_values("Probabilitas (%)", ascending=False)
        st.bar_chart(probabilities, x="Tipe Akomodasi", y="Probabilitas (%)")
        st.dataframe(probabilities, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
