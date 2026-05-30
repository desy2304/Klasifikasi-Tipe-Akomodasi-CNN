import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import tensorflow as tf

from tensorflow.keras.models import load_model

ARTIFACTS = {
    'model': 'cnn_airbnb_model.h5',
    'scaler': 'scaler.pkl',
    'label_encoder': 'label_encoder_cnn.pkl',
    'feature_columns': 'feature_columns.pkl'
}


def load_artifacts():
    artifacts = {}
    missing = []
    for k, fname in ARTIFACTS.items():
        if os.path.exists(fname):
            try:
                if k == 'model':
                    artifacts[k] = load_model(fname)
                else:
                    artifacts[k] = joblib.load(fname)
            except Exception as e:
                st.warning(f"Gagal memuat {fname}: {e}")
                missing.append(fname)
        else:
            missing.append(fname)
    return artifacts, missing


def save_uploaded_file(uploaded, target_path):
    if uploaded is None:
        return False
    bytes_data = uploaded.getvalue()
    with open(target_path, 'wb') as f:
        f.write(bytes_data)
    return True


def klasifikasi_akomodasi_streamlit(listing_baru, artifacts):
    model = artifacts['model']
    scaler = artifacts['scaler']
    label_encoder = artifacts['label_encoder']
    feature_columns = artifacts['feature_columns']

    input_df = pd.DataFrame([listing_baru])

    # Ensure one-hot columns present (neighbourhood group/neighbourhood naming from notebook)
    input_encoded = pd.get_dummies(input_df, columns=[c for c in ['neighbourhood group', 'neighbourhood'] if c in input_df.columns])

    # Add missing columns
    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Reorder
    input_encoded = input_encoded[feature_columns]

    # Numeric columns are those present in scaler.feature_names_in_ if available
    cols_to_scale = [c for c in input_encoded.columns if c in getattr(scaler, 'feature_names_in_', input_encoded.columns)]
    if cols_to_scale:
        input_encoded[cols_to_scale] = scaler.transform(input_encoded[cols_to_scale])

    X_input = input_encoded.values.astype(np.float32)
    X_input = X_input.reshape(X_input.shape[0], X_input.shape[1], 1)

    pred_proba = model.predict(X_input, verbose=0)[0]
    pred_class = int(np.argmax(pred_proba))
    pred_label = label_encoder.inverse_transform([pred_class])[0]
    confidence = float(pred_proba[pred_class]) * 100.0

    return {
        'tipe_akomodasi': pred_label,
        'kepercayaan': confidence,
        'probabilitas': {label_encoder.classes_[i]: float(pred_proba[i]) for i in range(len(label_encoder.classes_))}
    }


def main():
    st.set_page_config(page_title="Klasifikasi Tipe Akomodasi", page_icon=":house:")
    st.title("Klasifikasi Tipe Akomodasi - Streamlit")

    st.markdown(
        "Aplikasi ini membutuhkan file model dan preprocessor:\n"
        "- `cnn_airbnb_model.h5`\n"
        "- `scaler.pkl`\n"
        "- `label_encoder_cnn.pkl`\n"
        "- `feature_columns.pkl`\n\n"
        "Jika belum ada, upload file-file tersebut di sidebar."
    )

    artifacts, missing = load_artifacts()

    with st.sidebar:
        st.header("Artifacts / Model Files")
        if missing:
            st.warning(f"File hilang: {', '.join(missing)}")
            for m in missing:
                uploaded = st.file_uploader(f"Upload {m}", key=m)
                if uploaded is not None:
                    saved = save_uploaded_file(uploaded, m)
                    if saved:
                        st.success(f"Tersimpan: {m} - silakan muat ulang aplikasi (Refresh)")
        else:
            st.success("Semua artefak ditemukan dan siap digunakan.")

    if missing:
        st.info("Setelah mengupload file yang hilang, tekan refresh halaman atau deploy ulang.")
        st.stop()

    # Build simple input form based on notebook minimal fields
    st.sidebar.header("Input Listing")
    selected_group = st.sidebar.text_input("Neighbourhood Group", value="Manhattan")
    selected_hood = st.sidebar.text_input("Neighbourhood", value="Chelsea")
    harga = st.sidebar.number_input("Harga per malam (USD)", value=100.0, min_value=0.0)
    min_nights = st.sidebar.number_input("Minimum nights", value=1, min_value=1)

    if st.sidebar.button("Lakukan Klasifikasi"):
        listing_user = {
            'neighbourhood group': selected_group,
            'neighbourhood': selected_hood,
            'price': harga,
            'minimum nights': min_nights,
            # Fallback numeric defaults (tweak as needed)
            'number of reviews': 30,
            'reviews per month': 0.5,
            'availability 365': 200,
            'calculated host listings count': 1,
            'latitude': 40.7128,
            'longitude': -74.0060
        }

        with st.spinner('Melakukan klasifikasi...'):
            hasil = klasifikasi_akomodasi_streamlit(listing_user, artifacts)

        st.subheader("Hasil")
        st.metric("Tipe Akomodasi", hasil['tipe_akomodasi'], delta=f"{hasil['kepercayaan']:.2f}%")
        st.write("Probabilitas per kelas:")
        st.json(hasil['probabilitas'])


if __name__ == '__main__':
    main()
