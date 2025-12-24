# ML_Heart Disease Prediction

Aplikasi **machine learning berbasis Streamlit** untuk memprediksi risiko penyakit jantung menggunakan **XGBoost** serta menyediakan **evaluasi model** dan **explainability** berbasis **SHAP**.

Project ini dibuat untuk tujuan edukasi dan analisis data, khususnya dalam memahami bagaimana model machine learning mengambil keputusan pada data klinis.

---

## Fitur Utama

- **Prediksi Risiko Penyakit Jantung**
  Menghasilkan prediksi biner dan probabilitas risiko.

- **Model Evaluation**
  Menampilkan:

  - Accuracy
  - ROC-AUC
  - Classification Report
  - Confusion Matrix
  - ROC Curve

- **Explainability (SHAP)**

  - Global feature importance
  - Penjelasan lokal per data (waterfall plot)
  - Menampilkan nilai asli setiap fitur untuk cross-check

- **Konsistensi Data**
  Aplikasi menggunakan dataset yang sama dengan proses training model (tidak perlu upload data).

---

## Struktur Project (Ringkas)

```
project_root/
├── data/              # Dataset
├── model/             # Model terlatih (xgb_pipeline.pkl)
├── notebook/          # Notebook eksplorasi & training
├── streamlit/
│   └── app.py         # Aplikasi Streamlit
├── requirements.txt
└── README.md
```

Notebook digunakan untuk memahami:

- struktur dataset,
- preprocessing,
- training dan tuning model,
- serta evaluasi awal sebelum deployment.

---

## Instalasi

Pastikan Python 3.9+ terinstal.

Install dependency:

```bash
pip install -r requirements.txt
```

---

## Menjalankan Aplikasi

Masuk ke folder Streamlit:

```bash
cd streamlit
```

Jalankan aplikasi:

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser:

```
http://localhost:8501
```

---
