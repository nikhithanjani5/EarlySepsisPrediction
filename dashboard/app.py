import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

# -----------------------------
# Page settings
# -----------------------------
st.set_page_config(
    page_title="Early Sepsis Prediction Dashboard",
    layout="wide"
)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "Dataset.csv"
RESULTS_DIR = BASE_DIR / "results"
MODEL_PATH = RESULTS_DIR / "model_comparison.csv"
MISSING_PATH = RESULTS_DIR / "feature_missingness.csv"

# -----------------------------
# Load files
# -----------------------------
df = pd.read_csv(DATA_PATH)
model_df = pd.read_csv(MODEL_PATH)
missing_df = pd.read_csv(MISSING_PATH)

# -----------------------------
# Dataset overview values
# -----------------------------
total_rows = len(df)
total_patients = df["Patient_ID"].nunique()
row_prevalence = df["SepsisLabel"].mean() * 100
patient_prevalence = df.groupby("Patient_ID")["SepsisLabel"].max().mean() * 100
iculos_min = df["ICULOS"].min()
iculos_max = df["ICULOS"].max()

# -----------------------------
# Title
# -----------------------------
st.title("Early Prediction of Sepsis Using Machine Learning and Deep Learning")
st.markdown(
    "This dashboard summarises dataset characteristics, time-series patterns, and model performance for the dissertation project."
)

# -----------------------------
# Dataset overview
# -----------------------------
st.subheader("Dataset Overview")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Total Rows", f"{total_rows:,}")
c2.metric("Unique Patients", f"{total_patients:,}")
c3.metric("Row-level Sepsis %", f"{row_prevalence:.2f}%")
c4.metric("Patient-level Sepsis %", f"{patient_prevalence:.2f}%")
c5.metric("ICULOS Range", f"{iculos_min} - {iculos_max}")

# -----------------------------
# EDA Visualisations
# -----------------------------
st.subheader("Exploratory Data Analysis")

# Class imbalance and ICULOS histogram
col1, col2 = st.columns(2)

class_counts = df["SepsisLabel"].value_counts().sort_index().reset_index()
class_counts.columns = ["SepsisLabel", "Count"]
class_counts["SepsisLabel"] = class_counts["SepsisLabel"].map({0: "Non-Sepsis", 1: "Sepsis"})

fig_class = px.bar(
    class_counts,
    x="SepsisLabel",
    y="Count",
    title="Class Distribution (Row Level)",
    text="Count"
)
col1.plotly_chart(fig_class, use_container_width=True)

fig_iculos = px.histogram(
    df,
    x="ICULOS",
    nbins=50,
    title="Distribution of ICU Length of Stay (ICULOS)"
)
col2.plotly_chart(fig_iculos, use_container_width=True)

st.markdown("""
**Interpretation:**  
- The class distribution shows severe imbalance, with sepsis cases representing a very small proportion of all time steps.  
- This imbalance is one of the main reasons why **AUPRC is especially important** in model evaluation.  
- The ICULOS histogram shows variation in ICU stay duration, which supports the time-series nature of the prediction task.
""")

# -----------------------------
# Missingness chart
# -----------------------------
st.subheader("Top Missing Features")

top_missing = missing_df.sort_values("MissingPercent", ascending=False).head(15)

fig_missing = px.bar(
    top_missing,
    x="MissingPercent",
    y="Feature",
    orientation="h",
    title="Top 15 Features by Missing Percentage",
    text="MissingPercent"
)

fig_missing.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig_missing, use_container_width=True)

st.markdown("""
**Interpretation:**  
- Many laboratory and blood-gas variables have very high missingness.  
- This supports the preprocessing decision to use **forward filling** and **missingness indicator features**.  
- Missingness itself may contain clinical information because measurements are not recorded uniformly across all patients and time steps.
""")

# -----------------------------
# Time-series trend plot
# -----------------------------
st.subheader("Time-Series Trend Visualisation")

possible_features = [
    col for col in ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
    if col in df.columns
]

if len(possible_features) > 0:
    selected_feature = st.selectbox("Select a physiological feature", possible_features)

    trend_df = (
        df[[selected_feature, "ICULOS", "SepsisLabel"]]
        .dropna()
        .groupby(["ICULOS", "SepsisLabel"], as_index=False)[selected_feature]
        .mean()
    )

    trend_df["SepsisLabel"] = trend_df["SepsisLabel"].map({0: "Non-Sepsis", 1: "Sepsis"})

    fig_trend = px.line(
        trend_df,
        x="ICULOS",
        y=selected_feature,
        color="SepsisLabel",
        title=f"Mean {selected_feature} Across ICU Time"
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown(f"""
**Interpretation:**  
- This plot shows how **{selected_feature}** changes over ICU time for sepsis and non-sepsis cases.  
- It helps demonstrate the temporal structure of the dataset and supports the use of sequence-based models such as GRU and LSTM.  
- Even when deep learning models do not outperform classical models, temporal visualisations remain important for understanding the clinical data.
""")
else:
    st.warning("No expected physiological features were found in the dataset for time-series plotting.")

# -----------------------------
# Model performance charts
# -----------------------------
st.subheader("Model Performance Comparison")

col3, col4 = st.columns(2)

fig_auroc = px.bar(
    model_df.sort_values("Test_AUROC", ascending=False),
    x="Model",
    y="Test_AUROC",
    title="Test AUROC by Model",
    text="Test_AUROC"
)
fig_auroc.update_xaxes(tickangle=25)
col3.plotly_chart(fig_auroc, use_container_width=True)

fig_auprc = px.bar(
    model_df.sort_values("Test_AUPRC", ascending=False),
    x="Model",
    y="Test_AUPRC",
    title="Test AUPRC by Model",
    text="Test_AUPRC"
)
fig_auprc.update_xaxes(tickangle=25)
col4.plotly_chart(fig_auprc, use_container_width=True)

st.markdown("""
**Interpretation:**  
- **XGBoost** achieves the best overall discrimination on the test set.  
- **Logistic Regression** performs strongly as an interpretable baseline.  
- Among the deep learning models, **LSTM Direct** performs better than the GRU-based approaches in the current experiments.  
- Because the dataset is highly imbalanced, **AUPRC should be given special attention alongside AUROC**.
""")

# -----------------------------
# Results table
# -----------------------------
st.subheader("Final Results Table")
st.dataframe(model_df, use_container_width=True)

# -----------------------------
# Key findings
# -----------------------------
st.subheader("Key Findings")

best_auroc_model = model_df.loc[model_df["Test_AUROC"].idxmax(), "Model"]
best_auprc_model = model_df.loc[model_df["Test_AUPRC"].idxmax(), "Model"]

st.markdown(f"""
- **Best AUROC:** {best_auroc_model}  
- **Best AUPRC:** {best_auprc_model}  
- **XGBoost currently performs best overall on the test set.**  
- **AUPRC is especially important because the dataset is highly imbalanced.**  
- **Logistic Regression remains a strong interpretable baseline.**  
- **LSTM Direct is the strongest deep learning model among the current experiments.**
""")