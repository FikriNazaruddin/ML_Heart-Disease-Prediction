import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    layout="wide"
)

# =============================
# Load Model (Pipeline)
# =============================
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "..", "model", "xgb_pipeline.pkl")
    return joblib.load(model_path)

# =============================
# Load Training Dataset
# =============================
@st.cache_data
def load_training_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "data", "heart.csv")
    df = pd.read_csv(data_path)

    # Konsisten dengan notebook
    if "target" not in df.columns and "num" in df.columns:
        df = df.copy()
        df["target"] = (df["num"] > 0).astype(int)

    return df


pipeline = load_model()
preprocessor = pipeline.named_steps["preprocess"]
model = pipeline.named_steps["model"]

df = load_training_data()

# =============================
# UI
# =============================
st.title("Prediksi Risiko Penyakit Jantung")
st.write(
    "Aplikasi ini memprediksi risiko penyakit jantung menggunakan "
    "model XGBoost dan menyediakan evaluasi serta explainability berbasis SHAP."
)

menu = st.sidebar.radio(
    "Menu",
    ["Home", "Prediction", "Explainability", "Model Evaluation"]
)

# =============================
# HOME
# =============================
if menu == "Home":
    st.subheader("Tentang Proyek")
    st.write(""" 
    Model ini dilatih menggunakan dataset klinis pasien penyakit jantung
    dengan pendekatan machine learning berbasis XGBoost.

    Aplikasi ini menampilkan:
    - hasil prediksi risiko,
    - evaluasi performa model,
    - serta penjelasan keputusan model menggunakan SHAP.

    Catatan: Aplikasi ini hanya untuk tujuan edukasi.
    """)

# =============================
# PREDICTION
# =============================
elif menu == "Prediction":
    st.subheader("Masukkan Data Pasien untuk Prediksi Risiko Penyakit Jantung")

    # Penjelasan singkat tentang tiap fitur
    feature_explanations = {
        "age": "Usia pasien (tahun). Misalnya, 45 untuk usia 45 tahun.",
        "sex": "Jenis kelamin pasien. Pilih 'Male' untuk pria, 'Female' untuk wanita.",
        "cp": "Tipe nyeri dada: 'typical angina', 'atypical angina', 'non-anginal', atau 'asymptomatic'.",
        "trestbps": "Tekanan darah saat istirahat (dalam mmHg), misalnya 120.",
        "chol": "Kadar kolesterol serum (mg/dl), misalnya 250.",
        "fbs": "Apakah gula darah puasa lebih dari 120 mg/dl? Pilih 'True' atau 'False'.",
        "restecg": "Hasil EKG saat istirahat: 'normal', 'st-t abnormality', atau 'lv hypertrophy'.",
        "thalch": "Denyut jantung maksimum tercapai (dalam bpm). Misalnya 140.",
        "exang": "Angina yang dipicu oleh olahraga: Pilih 'True' atau 'False'.",
        "oldpeak": "Depresi ST akibat olahraga. Misalnya, 1.5 jika terjadi depresi ST 1.5mm.",
        "slope": "Kemiringan segmen ST saat puncak olahraga: 'upsloping', 'flat', atau 'downsloping'.",
        "ca": "Jumlah pembuluh darah utama yang terlihat (dalam angka 0-3).",
        "thal": "Status thalassemia: 'normal', 'fixed defect', atau 'reversable defect'."
    }

    # Membuat form untuk input data pasien
    with st.form(key="prediction_form"):
        # Membuat input untuk setiap fitur dengan opsi kosong
        st.write("#### Fitur: Usia")
        age = st.number_input("Usia Pasien (age)", min_value=1, max_value=120, value=None)

        st.write("#### Fitur: Jenis Kelamin")
        sex = st.selectbox("Jenis Kelamin (sex)", options=[None, "Male", "Female"])

        st.write("#### Fitur: Tipe Nyeri Dada")
        cp = st.selectbox("Tipe Nyeri Dada (cp)", options=[None, "typical angina", "atypical angina", "non-anginal", "asymptomatic"])

        st.write("#### Fitur: Tekanan Darah Istirahat")
        trestbps = st.number_input("Tekanan Darah Istirahat (trestbps)", min_value=80, max_value=200, value=None)

        st.write("#### Fitur: Kadar Kolesterol Serum")
        chol = st.number_input("Kadar Kolesterol Serum (chol)", min_value=100, max_value=600, value=None)

        st.write("#### Fitur: Gula Darah Puasa > 120 mg/dl")
        fbs = st.selectbox("Gula Darah Puasa > 120 mg/dl (fbs)", options=[None, "True", "False"])

        st.write("#### Fitur: Hasil EKG saat Istirahat")
        restecg = st.selectbox("Hasil EKG Saat Istirahat (restecg)", options=[None, "normal", "st-t abnormality", "lv hypertrophy"])

        st.write("#### Fitur: Denyut Jantung Maksimum Tercapai")
        thalch = st.number_input("Denyut Jantung Maksimum Tercapai (thalch)", min_value=50, max_value=220, value=None)

        st.write("#### Fitur: Angina yang Dipicu oleh Olahraga")
        exang = st.selectbox("Angina yang Dipicu oleh Olahraga (exang)", options=[None, "True", "False"])

        st.write("#### Fitur: Depresi ST Akibat Olahraga")
        oldpeak = st.number_input("Depresi ST Akibat Olahraga (oldpeak)", min_value=0.0, max_value=10.0, value=None)

        st.write("#### Fitur: Kemiringan Segmen ST saat Puncak Olahraga")
        slope = st.selectbox("Kemiringan Segmen ST saat Puncak Olahraga (slope)", options=[None, "upsloping", "flat", "downsloping"])

        st.write("#### Fitur: Jumlah Pembuluh Darah Utama yang Terlihat")
        ca = st.number_input("Jumlah Pembuluh Darah Utama yang Terlihat (ca)", min_value=0, max_value=3, value=None)

        st.write("#### Fitur: Status Thalassemia")
        thal = st.selectbox("Status Thalassemia (thal)", options=[None, "normal", "fixed defect", "reversable defect"])

        # Submit button
        submit_button = st.form_submit_button(label="Prediksi")

    # Mengonversi data input menjadi DataFrame agar bisa diproses model
    if submit_button:
        user_data = {
            "age": age if age is not None else np.nan,
            "sex": sex if sex is not None else np.nan,
            "cp": cp if cp is not None else np.nan,
            "trestbps": trestbps if trestbps is not None else np.nan,
            "chol": chol if chol is not None else np.nan,
            "fbs": fbs if fbs is not None else np.nan,
            "restecg": restecg if restecg is not None else np.nan,
            "thalch": thalch if thalch is not None else np.nan,
            "exang": exang if exang is not None else np.nan,
            "oldpeak": oldpeak if oldpeak is not None else np.nan,
            "slope": slope if slope is not None else np.nan,
            "ca": ca if ca is not None else np.nan,
            "thal": thal if thal is not None else np.nan
        }

        input_df = pd.DataFrame([user_data])

        # Lakukan preprocessing pada data input
        input_transformed = preprocessor.transform(input_df)

        # Prediksi hasil dan probabilitas
        prediction = model.predict(input_transformed)[0]
        probability = model.predict_proba(input_transformed)[:, 1][0]

        # Menampilkan hasil prediksi
        st.write("#### Hasil Prediksi Model")
        st.write(f"**Prediksi Penyakit Jantung:** {'Penyakit Jantung' if prediction == 1 else 'Tidak Ada Penyakit Jantung'}")
        st.write(f"**Probabilitas Risiko Penyakit Jantung:** {probability:.3f}")

        # ============================
        # SHAP Waterfall Plot
        # ============================
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_transformed)

        st.markdown("#### Penjelasan Lokal (SHAP Waterfall)")

        fig_local, ax_local = plt.subplots()
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=input_transformed[0],
                feature_names=preprocessor.get_feature_names_out()
            ),
            show=False
        )
        st.pyplot(fig_local)

# =============================
# EXPLAINABILITY (SHAP)
# =============================
elif menu == "Explainability":
    st.subheader("Explainability dengan SHAP")

    X = df.drop(columns=["target"], errors="ignore")

    # =============================
    # GLOBAL EXPLANATION
    # =============================
    st.markdown("### Global Feature Importance")

    X_transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    fig_global, ax_global = plt.subplots()
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        show=False
    )
    st.pyplot(fig_global)

    st.markdown("---")

    # =============================
    # LOCAL EXPLANATION
    # =============================
    st.markdown("### Penjelasan Prediksi Individu")

    idx = st.number_input(
        "Pilih index data",
        min_value=0,
        max_value=len(df) - 1,
        value=0
    )

    selected_row = df.iloc[[idx]]

    st.markdown("#### Data Asli (Raw Features)")
    raw_features = selected_row.T
    raw_features.columns = ["Value"]
    st.dataframe(raw_features)

    st.markdown("#### Target Asli & Hasil Prediksi")

    st.write(f"Target Asli (Binary): {int(df.iloc[idx]['target'])}")

    X_single = selected_row.drop(columns=["target"], errors="ignore")
    pred = pipeline.predict(X_single)[0]
    prob = pipeline.predict_proba(X_single)[0, 1]

    st.write(f"Prediksi Model: {pred}")
    st.write(f"Probabilitas Risiko: {prob:.3f}")

    st.markdown("#### Penjelasan Lokal (SHAP Waterfall)")

    X_single_transformed = preprocessor.transform(X_single)
    shap_single = explainer.shap_values(X_single_transformed)

    fig_local, ax_local = plt.subplots()
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_single[0],
            base_values=explainer.expected_value,
            data=X_single_transformed[0],
            feature_names=feature_names
        ),
        show=False
    )
    st.pyplot(fig_local)

# =============================
# MODEL EVALUATION
# =============================
elif menu == "Model Evaluation":
    st.subheader("Evaluasi Model XGBoost")

    X = df.drop(columns=["target"], errors="ignore")
    y_true = df["target"]

    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]

    acc = accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_proba)

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{acc:.3f}")
    col2.metric("ROC-AUC", f"{roc:.3f}")

    st.markdown("### Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)

    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
        ax=ax_cm
    )
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    st.markdown("### ROC Curve")
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"AUC = {roc:.3f}")
    ax_roc.plot([0, 1], [0, 1], linestyle="--")
    ax_roc.legend()
    st.pyplot(fig_roc)
