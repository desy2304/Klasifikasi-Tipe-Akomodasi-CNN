# Klasifikasi Tipe Akomodasi - Streamlit App

Langkah cepat untuk deploy ke Streamlit Community Cloud:

1. Pastikan repository berisi file berikut di root:
   - `app.py` (aplikasi Streamlit)
   - `requirements.txt` (dependensi)
   - model & artefak: `cnn_airbnb_model.h5`, `scaler.pkl`, `label_encoder_cnn.pkl`, `feature_columns.pkl`

2. Jika model terlalu besar untuk di-push ke GitHub, simpan model di storage eksternal (S3/Google Drive) dan ubah `app.py` agar mendownload model saat startup.

3. Buat repository di GitHub dan push semua file (`git init` / `git remote add origin ...` / `git push`).

4. Buka https://share.streamlit.io -> "New app" -> pilih repository, branch, dan `app.py` sebagai main file -> Deploy.

5. Klik "Advanced settings" dan pilih Python 3.11 sebelum deploy. Ini penting karena TensorFlow 2.12 tidak cocok dengan default Python terbaru di Streamlit Cloud.

6. Jika aplikasi membutuhkan secret atau URL untuk model, atur environment variables di Streamlit Cloud (Settings -> Secrets).

Catatan:
- Jika Anda ingin saya menambahkan fungsi download model otomatis dari URL, beri tahu URL model Anda atau izinkan saya membuat placeholder untuk `MODEL_URL`.
- Untuk pengujian lokal: jalankan `streamlit run app.py` di direktori project.
