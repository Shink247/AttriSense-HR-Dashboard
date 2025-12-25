import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- Config --------------------
st.set_page_config(page_title="AttriSense HR Dashboard", page_icon="🤖", layout="wide")
st.markdown("""
<style>
    .main {
        background-color: #f5e6d3;
    }
    [data-testid="stSidebar"] {
        background-color: #c1834c !important;
    }
    .stApp {
        background-color: #f5e6d3;
    }
</style>
""", unsafe_allow_html=True)

BROWN = "#5A2C00"
BLACK = "#000000"
GREEN = "#228B22"
RED = "#B22222"

# -------------------- Styling Functions --------------------
def brown_title(text, size="h1"):
    st.markdown(f"<{size} style='color:{BROWN}; font-weight:700'>{text}</{size}>", unsafe_allow_html=True)

def brown_text(text):
    st.markdown(f"<p style='color:{BROWN}; font-size:18px'>{text}</p>", unsafe_allow_html=True)

def black_subheader(text):
    st.markdown(f"<h3 style='color:{BLACK}'>{text}</h3>", unsafe_allow_html=True)

# -------------------- Logo --------------------
try:
    logo = Image.open("attrisense-logo.png")
    st.sidebar.image(logo, width=120)
    st.image(logo, width=180)
except FileNotFoundError:
    st.sidebar.warning("⚠️ Logo not found")

# -------------------- Sidebar --------------------
# Custom Sidebar Layout with Sticky Footer
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100vh;
    }

    .sidebar-footer {
        text-align: center;
        font-size: 14px;
        color: white;
        padding-bottom: 15px;
    }

    .sidebar-footer hr {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        border: none;
        height: 1px;
        background-color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar content
with st.sidebar:
    st.title("🤖 AttriSense HR Dashboard")
    st.write("Enterprise AI-powered people analytics")
    st.markdown("---")

    selected_page = st.radio(
        "Navigation",
        ["🏠 Home", "📝 Upload Data", "📊 Dashboard", "🧠 Attrition Insights",
         "📈 Visualizations", "📂 Employee Profiles", "⚙️ Settings"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("🔒 Secure Enterprise AI SaaS")

    # Footer at bottom
    st.markdown("""
    <div class="sidebar-footer">
        <hr>
        👩‍💻 Made with 🤍 by <a href="https://www.linkedin.com/in/shinora-khan/" target="_blank" style="color:white;"><strong>Shinora Khan</strong></a>
    </div>
    """, unsafe_allow_html=True)

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    with open("attrition_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
expected_features = list(model.feature_names_in_)
raw_data = None

# -------------------- Preprocessing --------------------
def preprocess_input(df):
    df_encoded = pd.get_dummies(df, drop_first=False)
    df_encoded = df_encoded.reindex(columns=expected_features, fill_value=0)
    return df_encoded

# -------------------- Upload Data Page --------------------
if selected_page == "📝 Upload Data":
    brown_title("📂 Upload Employee CSV File", size="h2")
    brown_text("Upload your employee data to generate attrition predictions.")
    uploaded_file = st.file_uploader("", type="csv", label_visibility="collapsed")

    if uploaded_file:
        raw_data = pd.read_csv(uploaded_file)
        st.success("✅ Data uploaded successfully")
        st.dataframe(raw_data.head())

        try:
            processed_data = preprocess_input(raw_data)
            predictions = model.predict(processed_data)
            raw_data["Attrition_Prediction"] = predictions

            brown_title("📊 Prediction Results", size="h3")
            st.dataframe(raw_data)

            csv = raw_data.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Prediction Report", csv, "attrisense_predictions.csv", "text/csv")

            st.session_state["raw_data"] = raw_data

        except Exception as e:
            st.error("⚠️ Unable to generate predictions")
            st.code(str(e))

# -------------------- Home Page --------------------
elif selected_page == "🏠 Home":
    brown_title("Welcome to AttriSense HR Dashboard")
    brown_text("""
    AttriSense is an Enterprise‑grade AI People Analytics platform designed for modern HR teams.

    📊 Predict Attrition Before It Happens 
    Machine‑learning models identify high‑risk employees early.

    🧠 Explainable AI Insights  
    Understand *why* employees leave — not just who.

    📈 HR‑Focused Analytics
    Analyze departments, roles, travel patterns, and satisfaction trends.

    ⚙️ Plug‑and‑Play Simplicity 
    Upload your employee CSV — AttriSense handles the rest.

    Built for HR leaders, workforce planners, and data‑driven organizations. """)

# -------------------- Dashboard Page --------------------
elif selected_page == "📊 Dashboard":
    raw_data = st.session_state.get("raw_data")
    if raw_data is not None:
        brown_title("📊 Dashboard Overview", size="h2")

        total = len(raw_data)
        at_risk = raw_data['Attrition_Prediction'].sum()
        rate = (at_risk / total) * 100

        col1, col2, col3 = st.columns(3)

        col1.markdown(f"<p style='color:{BLACK}; font-size:20px;'>👥 Total Employees</p>", unsafe_allow_html=True)
        col1.markdown(f"<h1 style='color:{GREEN};'>{total}</h1>", unsafe_allow_html=True)

        col2.markdown(f"<p style='color:{BLACK}; font-size:20px;'>⚠️ At-Risk Employees</p>", unsafe_allow_html=True)
        col2.markdown(f"<h1 style='color:{RED};'>{at_risk}</h1>", unsafe_allow_html=True)

        col3.markdown(f"<p style='color:{BLACK}; font-size:20px;'>📉 Attrition Rate</p>", unsafe_allow_html=True)
        col3.markdown(f"<h1 style='color:{BLACK};'>{rate:.2f}%</h1>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please upload data in the 'Upload Data' tab.")

# -------------------- Visualizations --------------------
elif selected_page == "📈 Visualizations":
    raw_data = st.session_state.get("raw_data")
    if raw_data is not None:
        brown_title("📈 Visualizations", size="h2")

        # 1. Attrition by Department
        black_subheader("🎯 Attrition by Department")
        fig1, ax1 = plt.subplots()
        sns.countplot(x='Department', hue='Attrition_Prediction', data=raw_data, palette='pastel', ax=ax1)
        ax1.set_title("Department-wise Attrition", fontsize=12)
        st.pyplot(fig1)

        # 2. Satisfaction vs Attrition
        black_subheader("😊 Satisfaction vs Attrition")
        fig2, ax2 = plt.subplots()
        sns.boxplot(x='Attrition_Prediction', y='JobSatisfaction', data=raw_data, palette='Set3', ax=ax2)
        ax2.set_title("Job Satisfaction by Attrition", fontsize=12)
        st.pyplot(fig2)

        # 3. Pie Chart
        black_subheader("📊 Overall Attrition Breakdown")
        pie_data = raw_data['Attrition_Prediction'].value_counts()
        labels = ["Stayed", "Left"] if 0 in pie_data.index else ["Left", "Stayed"]
        fig3, ax3 = plt.subplots()
        ax3.pie(pie_data, labels=labels, autopct='%1.1f%%', colors=["#90ee90", "#ff7f7f"], startangle=90)
        ax3.axis("equal")
        st.pyplot(fig3)

        # 4. Heatmap (if numeric cols exist)
        black_subheader("🔥 Correlation Heatmap")
        numeric_cols = raw_data.select_dtypes(include='number')
        fig4, ax4 = plt.subplots()
        sns.heatmap(numeric_cols.corr(), cmap="YlOrBr", annot=True, fmt=".2f", ax=ax4)
        st.pyplot(fig4)

    else:
        st.warning("⚠️ Please upload data in the 'Upload Data' tab.")

# -------------------- Attrition Insights --------------------
elif selected_page == "🧠 Attrition Insights":
    brown_title("🧠 Attrition Insights", size="h2")
    raw_data = st.session_state.get("raw_data")

    if raw_data is not None:
        # Check if the model supports feature importance
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_names = expected_features
            feature_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
            feature_df = feature_df.sort_values(by="Importance", ascending=True)

            brown_text("🔍 Top Predictors of Attrition (Model-Based):")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x="Importance", y="Feature", data=feature_df, palette="YlOrBr", ax=ax)
            ax.set_title("Feature Importance", fontsize=14)
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")
            st.pyplot(fig)
        
        else:
            # Fallback to correlation-based analysis
            brown_text("📊 Correlation with Attrition (Fallback Insights):")
            corr_df = raw_data.copy()
            corr_df['Attrition'] = corr_df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

            # Only use numeric features
            numeric_cols = corr_df.select_dtypes(include='number').columns
            correlations = corr_df[numeric_cols].corr()['Attrition'].drop('Attrition').sort_values()

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x=correlations.values, y=correlations.index, palette="coolwarm", ax=ax)
            ax.set_title("Feature Correlation with Attrition", fontsize=14)
            ax.set_xlabel("Correlation")
            ax.set_ylabel("Feature")
            st.pyplot(fig)
    else:
        st.warning("⚠️ Please upload data in the 'Upload Data' tab to view attrition insights.")

# -------------------- Employee Profiles --------------------
elif selected_page == "📂 Employee Profiles":
    brown_title("👥 Employee Profiles", size="h2")
    raw_data = st.session_state.get("raw_data")
    if raw_data is not None:
        search_term = st.text_input("🔍 Search by Employee ID, Name, or Department:")
        if search_term:
            filtered_data = raw_data[raw_data.apply(lambda row: search_term.lower() in str(row).lower(), axis=1)]
            st.dataframe(filtered_data, use_container_width=True)
        else:
            st.dataframe(raw_data, use_container_width=True)
    else:
        st.warning("⚠️ Please upload data in the 'Upload Data' tab.")

# -------------------- Settings --------------------
elif selected_page == "⚙️ Settings":
    brown_title("⚙️ Settings", size="h2")
    brown_text("Customize model thresholds and real-time alerts.")
    threshold = st.slider("Set attrition alert threshold", 0.0, 1.0, 0.5)
    st.session_state["alert_threshold"] = threshold
    st.success(f"🔧 Alerts will trigger for attrition probability above {threshold}")
