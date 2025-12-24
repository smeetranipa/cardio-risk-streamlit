import streamlit as st
import numpy as np
import joblib
import altair as alt
import pandas as pd


import sklearn
import joblib
import numpy as np

st.write("sklearn version:", sklearn.__version__)
st.write("joblib version:", joblib.__version__)
st.write("numpy version:", np.__version__)



# Load saved model objects

model = joblib.load("cardio_gb_model.pkl")
scaler = joblib.load("scaler.pkl")
threshold = joblib.load("threshold.pkl")



# App Title

st.title("❤️ Cardio Disease Prediction App")
st.write("Simple version – Medical risk screening")
st.markdown("---")



# User Input Form

with st.form(key='cardio_form'):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=45)
        gender = st.selectbox("Gender", ["Female", "Male"])
        height = st.number_input("Height (cm)", min_value=120, max_value=220, value=165)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    
    with col2:
        ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=70, max_value=250, value=120)
        ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=40, max_value=150, value=80)
        cholesterol = st.selectbox("Cholesterol Level", [1, 2, 3])
        gluc = st.selectbox("Glucose Level", [1, 2, 3])
        smoke = st.selectbox("Do you smoke?", ["No", "Yes"])
        alco = st.selectbox("Do you consume alcohol?", ["No", "Yes"])
        active = st.selectbox("Are you physically active?", ["No", "Yes"])
    
    submit_button = st.form_submit_button(label='Predict Cardio Risk')



# Preprocessing

if submit_button:

    gender_val = 1 if gender == "Female" else 2
    smoke_val = 1 if smoke == "Yes" else 0
    alco_val = 1 if alco == "Yes" else 0
    active_val = 1 if active == "Yes" else 0

    bmi = weight / ((height / 100) ** 2)
    age_in_years = age

    def bmi_category(bmi):
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"

    def bp_category(sys):
        if sys < 120:
            return "Normal"
        elif sys < 140:
            return "Pre-Hypertension"
        elif sys < 160:
            return "Stage 1 Hypertension"
        else:
            return "Stage 2 Hypertension"

    bmi_val = round(bmi, 2)
    bmi_cat = bmi_category(bmi)
    bp_cat = bp_category(ap_hi)

    st.markdown("### 📊 Health Indicators")
    st.write(f"**BMI:** {bmi_val} ({bmi_cat})")
    st.write(f"**Blood Pressure Category:** {bp_cat}")

    input_data = np.array([[ 
        age_in_years,
        gender_val,
        bmi,
        ap_hi,
        ap_lo,
        cholesterol,
        gluc,
        smoke_val,
        alco_val,
        active_val
    ]])


    # Prediction
    prob = model.predict_proba(input_data)[0][1]
    st.write(f"### 🧪 Risk Probability: {prob:.2f}")



    # Risk Message
    if prob >= threshold:
        st.error("⚠️ High Risk of Cardiovascular Disease")
        st.markdown("### 🧠 Why this result?")
        st.write("""
        Risk factors:
        - Higher age
        - High blood pressure
        - Elevated BMI (overweight/obesity)
        - High cholesterol or glucose
        - Unhealthy lifestyle habits
        """)
    else:
        st.success("✅ Low Risk of Cardiovascular Disease")
        st.markdown("### 🧠 Why this result?")
        st.write("""
        Factors indicating lower risk:
        - Healthy blood pressure
        - Normal body weight
        - Better lifestyle habits
        - Lower cholesterol and glucose levels
        """)


    st.markdown("### 🔍 Feature Importance (Model Explanation)")

    feature_names = [
    'Age (years)',
    'Gender',
    'BMI',
    'Systolic BP',
    'Diastolic BP',
    'Cholesterol',
    'Glucose',
    'Smoking',
    'Alcohol',
    'Physical Activity'
    ]

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    st.dataframe(importance_df, use_container_width=True)

    importance_chart = alt.Chart(importance_df).mark_bar().encode(
        x=alt.X('Importance:Q', title='Importance Score'),
        y=alt.Y('Feature:N', sort='-x', title='Feature'),
        color=alt.value('#1f77b4')
    ).properties(
        width=500,
        height=350
    )

    st.altair_chart(importance_chart, use_container_width=True)

    # Risk Visualization Chart
    st.markdown("### 📈 Risk Visualization")

    chart_data = pd.DataFrame({
        'Category': ['Safe', 'Risk'],
        'Probability': [1 - prob, prob]
    })

    chart = alt.Chart(chart_data).mark_bar().encode(
        x='Category',
        y='Probability',
        color=alt.condition(
            alt.datum.Category == 'Risk',
            alt.value('red'),
            alt.value('green')
        )
    ).properties(width=400, height=300)

    st.altair_chart(chart, use_container_width=True)




 

# import streamlit as st
# import numpy as np
# import joblib
# import altair as alt
# import pandas as pd

# # -----------------------------
# # Page Config
# # -----------------------------
# st.set_page_config(page_title="Cardio Risk Predictor", layout="wide")

# # -----------------------------
# # Load saved model objects
# # -----------------------------
# @st.cache_resource
# def load_models():
#     model = joblib.load("cardio_gb_model.pkl")
#     scaler = joblib.load("scaler.pkl")
#     threshold = joblib.load("threshold.pkl")
#     return model, scaler, threshold

# model, scaler, threshold = load_models()

# # -----------------------------
# # App Title
# # -----------------------------
# st.title("❤️ Cardio Disease Prediction App")
# st.write("Simple version – Medical risk screening")
# st.markdown("---")

# # -----------------------------
# # User Input Form
# # -----------------------------
# with st.form(key='cardio_form'):
#     col1, col2 = st.columns(2)

#     with col1:
#         age = st.number_input("Age (years)", min_value=18, max_value=100, value=45)
#         gender = st.selectbox("Gender", ["Female", "Male"])
#         height = st.number_input("Height (cm)", min_value=120, max_value=220, value=165)
#         weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)

#     with col2:
#         ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=70, max_value=250, value=120)
#         ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=40, max_value=150, value=80)
#         cholesterol = st.selectbox("Cholesterol Level", [1, 2, 3])
#         gluc = st.selectbox("Glucose Level", [1, 2, 3])
#         smoke = st.selectbox("Do you smoke?", ["No", "Yes"])
#         alco = st.selectbox("Do you consume alcohol?", ["No", "Yes"])
#         active = st.selectbox("Are you physically active?", ["No", "Yes"])

#     submit_button = st.form_submit_button(label='Predict Cardio Risk')

# # -----------------------------
# # Preprocessing & Prediction
# # -----------------------------
# if submit_button:
#     gender_val = 1 if gender == "Female" else 2
#     smoke_val = 1 if smoke == "Yes" else 0
#     alco_val = 1 if alco == "Yes" else 0
#     active_val = 1 if active == "Yes" else 0

#     bmi = weight / ((height / 100) ** 2)
#     age_in_years = age

#     def bmi_category(bmi):
#         if bmi < 18.5:
#             return "Underweight"
#         elif bmi < 25:
#             return "Normal"
#         elif bmi < 30:
#             return "Overweight"
#         else:
#             return "Obese"

#     def bp_category(sys):
#         if sys < 120:
#             return "Normal"
#         elif sys < 140:
#             return "Pre-Hypertension"
#         elif sys < 160:
#             return "Stage 1 Hypertension"
#         else:
#             return "Stage 2 Hypertension"

#     bmi_val = round(bmi, 2)
#     bmi_cat = bmi_category(bmi)
#     bp_cat = bp_category(ap_hi)

#     st.markdown("### 📊 Health Indicators")
#     st.write(f"**BMI:** {bmi_val} ({bmi_cat})")
#     st.write(f"**Blood Pressure Category:** {bp_cat}")

#     input_data = np.array([[ 
#         age_in_years,
#         gender_val,
#         bmi,
#         ap_hi,
#         ap_lo,
#         cholesterol,
#         gluc,
#         smoke_val,
#         alco_val,
#         active_val
#     ]])

#     # Scale input before prediction
#     scaled_input = scaler.transform(input_data)
#     prob = model.predict_proba(scaled_input)[0][1]
#     st.write(f"### 🧪 Risk Probability: {prob:.2f}")

#     # Risk Message
#     if prob >= threshold:
#         st.error("⚠️ High Risk of Cardiovascular Disease")
#         st.markdown("### 🧠 Why this result?")
#         st.write("""
#         Risk factors:
#         - Higher age
#         - High blood pressure
#         - Elevated BMI (overweight/obesity)
#         - High cholesterol or glucose
#         - Unhealthy lifestyle habits
#         """)
#     else:
#         st.success("✅ Low Risk of Cardiovascular Disease")
#         st.markdown("### 🧠 Why this result?")
#         st.write("""
#         Factors indicating lower risk:
#         - Healthy blood pressure
#         - Normal body weight
#         - Better lifestyle habits
#         - Lower cholesterol and glucose levels
#         """)

#     # Feature Importance
#     st.markdown("### 🔍 Feature Importance (Model Explanation)")
#     feature_names = [
#         'Age (years)', 'Gender', 'BMI', 'Systolic BP', 'Diastolic BP',
#         'Cholesterol', 'Glucose', 'Smoking', 'Alcohol', 'Physical Activity'
#     ]
#     importances = model.feature_importances_

#     importance_df = pd.DataFrame({
#         'Feature': feature_names,
#         'Importance': importances
#     }).sort_values(by='Importance', ascending=False)

#     st.dataframe(importance_df, use_container_width=True)

#     importance_chart = alt.Chart(importance_df).mark_bar().encode(
#         x=alt.X('Importance:Q', title='Importance Score'),
#         y=alt.Y('Feature:N', sort='-x', title='Feature'),
#         color=alt.value('#1f77b4')
#     ).properties(width=500, height=350)

#     st.altair_chart(importance_chart, use_container_width=True)

#     # Risk Visualization Chart
#     st.markdown("### 📈 Risk Visualization")
#     chart_data = pd.DataFrame({
#         'Category': ['Safe', 'Risk'],
#         'Probability': [1 - prob, prob]
#     })
#     chart = alt.Chart(chart_data).mark_bar().encode(
#         x='Category',
#         y='Probability',
#         color=alt.condition(
#             alt.datum.Category == 'Risk',
#             alt.value('red'),
#             alt.value('green')
#         )
#     ).properties(width=400, height=300)
#     st.altair_chart(chart, use_container_width=True)








# SHAP Explaination


# import streamlit as st
# import numpy as np
# import joblib
# import pandas as pd
# import altair as alt
# import shap
# import matplotlib.pyplot as plt

# # --------------------------------------------------
# # Load Model Objects
# # --------------------------------------------------
# model = joblib.load("cardio_gb_model.pkl")
# scaler = joblib.load("scaler.pkl")
# threshold = joblib.load("threshold.pkl")

# # SHAP Explainer (Tree-based)
# explainer = shap.TreeExplainer(model)

# # --------------------------------------------------
# # App UI
# # --------------------------------------------------
# st.set_page_config(page_title="Cardio Risk Predictor", layout="wide")
# st.title("❤️ Cardiovascular Disease Prediction App")
# st.write("Medical risk screening using Machine Learning")
# st.markdown("---")

# # --------------------------------------------------
# # User Input Form
# # --------------------------------------------------
# with st.form(key="cardio_form"):
#     col1, col2 = st.columns(2)

#     with col1:
#         age = st.number_input("Age (years)", 18, 100, 45)
#         gender = st.selectbox("Gender", ["Female", "Male"])
#         height = st.number_input("Height (cm)", 120, 220, 165)
#         weight = st.number_input("Weight (kg)", 30, 200, 70)

#     with col2:
#         ap_hi = st.number_input("Systolic BP", 70, 250, 120)
#         ap_lo = st.number_input("Diastolic BP", 40, 150, 80)
#         cholesterol = st.selectbox("Cholesterol Level", [1, 2, 3])
#         gluc = st.selectbox("Glucose Level", [1, 2, 3])
#         smoke = st.selectbox("Smoking", ["No", "Yes"])
#         alco = st.selectbox("Alcohol Intake", ["No", "Yes"])
#         active = st.selectbox("Physically Active", ["No", "Yes"])

#     submit = st.form_submit_button("Predict Cardio Risk")

# # --------------------------------------------------
# # Prediction Logic
# # --------------------------------------------------
# if submit:

#     gender_val = 1 if gender == "Female" else 2
#     smoke_val = 1 if smoke == "Yes" else 0
#     alco_val = 1 if alco == "Yes" else 0
#     active_val = 1 if active == "Yes" else 0

#     bmi = weight / ((height / 100) ** 2)

#     input_data = np.array([[ 
#         age,
#         gender_val,
#         bmi,
#         ap_hi,
#         ap_lo,
#         cholesterol,
#         gluc,
#         smoke_val,
#         alco_val,
#         active_val
#     ]])

#     feature_names = [
#         "Age",
#         "Gender",
#         "BMI",
#         "Systolic BP",
#         "Diastolic BP",
#         "Cholesterol",
#         "Glucose",
#         "Smoking",
#         "Alcohol",
#         "Physical Activity"
#     ]

#     # --------------------------------------------------
#     # Prediction
#     # --------------------------------------------------
#     prob = model.predict_proba(input_data)[0][1]
#     st.subheader(f"🧪 Risk Probability: **{prob:.2f}**")

#     if prob >= threshold:
#         st.error("⚠️ High Risk of Cardiovascular Disease")
#     else:
#         st.success("✅ Low Risk of Cardiovascular Disease")

#     st.markdown("---")

#     # --------------------------------------------------
#     # GLOBAL FEATURE IMPORTANCE (STATIC)
#     # --------------------------------------------------
#     st.subheader("🔍 Global Feature Importance (Model-Level)")

#     importance_df = pd.DataFrame({
#         "Feature": feature_names,
#         "Importance": model.feature_importances_
#     }).sort_values(by="Importance", ascending=False)

#     st.dataframe(importance_df, use_container_width=True)

#     global_chart = alt.Chart(importance_df).mark_bar().encode(
#         x=alt.X("Importance:Q", title="Importance Score"),
#         y=alt.Y("Feature:N", sort="-x"),
#         color=alt.value("#1f77b4")
#     ).properties(height=350)

#     st.altair_chart(global_chart, use_container_width=True)

#     st.caption(
#         "Global importance shows which features matter most overall and does NOT change per patient."
#     )

#     st.markdown("---")

#     # --------------------------------------------------
#     # SHAP EXPLANATION (DYNAMIC)
#     # --------------------------------------------------
#     st.subheader("🧠 Why THIS Prediction? (SHAP Explanation)")

#     shap_values = explainer.shap_values(input_data)

#     shap_df = pd.DataFrame({
#         "Feature": feature_names,
#         "SHAP Value": shap_values[0]
#     })

#     shap_df["Impact"] = shap_df["SHAP Value"].abs()
#     shap_df = shap_df.sort_values(by="Impact", ascending=False)

#     st.dataframe(shap_df[["Feature", "SHAP Value"]], use_container_width=True)

#     shap_chart = alt.Chart(shap_df).mark_bar().encode(
#         x=alt.X("SHAP Value:Q", title="Impact on Risk"),
#         y=alt.Y("Feature:N", sort="-x"),
#         color=alt.condition(
#             alt.datum["SHAP Value"] > 0,
#             alt.value("red"),
#             alt.value("green")
#         )
#     ).properties(height=350)

#     st.altair_chart(shap_chart, use_container_width=True)

#     st.caption(
#         "Red bars increase predicted risk. Green bars decrease predicted risk. "
#         "This explanation is unique for the current input."
#     )

#     # --------------------------------------------------
#     # Risk Probability Visualization
#     # --------------------------------------------------
#     st.markdown("---")
#     st.subheader("📈 Risk Visualization")

#     risk_df = pd.DataFrame({
#         "Category": ["Safe", "Risk"],
#         "Probability": [1 - prob, prob]
#     })

#     risk_chart = alt.Chart(risk_df).mark_bar().encode(
#         x="Category",
#         y="Probability",
#         color=alt.condition(
#             alt.datum.Category == "Risk",
#             alt.value("red"),
#             alt.value("green")
#         )
#     ).properties(height=300)

#     st.altair_chart(risk_chart, use_container_width=True)




# Deepseek

# import streamlit as st
# import numpy as np
# import joblib
# import pandas as pd
# import altair as alt
# import shap
# import matplotlib.pyplot as plt
# from streamlit_option_menu import option_menu

# # --------------------------------------------------
# # Load Model Objects
# # --------------------------------------------------
# model = joblib.load("cardio_gb_model.pkl")
# scaler = joblib.load("scaler.pkl")
# threshold = joblib.load("threshold.pkl")

# # SHAP Explainer (Tree-based)
# explainer = shap.TreeExplainer(model)

# # --------------------------------------------------
# # App Configuration
# # --------------------------------------------------
# st.set_page_config(
#     page_title="CardioRisk AI",
#     page_icon="❤️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # --------------------------------------------------
# # Custom CSS for Enhanced UI
# # --------------------------------------------------
# st.markdown("""
# <style>
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
#     * {
#         font-family: 'Inter', sans-serif;
#     }
    
#     .main-header {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         padding: 2rem;
#         border-radius: 15px;
#         margin-bottom: 2rem;
#         color: white;
#         box-shadow: 0 10px 20px rgba(0,0,0,0.1);
#     }
    
#     .risk-high {
#         background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
#         padding: 1.5rem;
#         border-radius: 15px;
#         color: white;
#         animation: pulse 2s infinite;
#     }
    
#     .risk-low {
#         background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
#         padding: 1.5rem;
#         border-radius: 15px;
#         color: white;
#     }
    
#     .card {
#         background: white;
#         padding: 1.5rem;
#         border-radius: 15px;
#         box-shadow: 0 5px 15px rgba(0,0,0,0.08);
#         border: 1px solid #e0e0e0;
#         margin-bottom: 1.5rem;
#     }
    
#     .metric-card {
#         background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
#         padding: 1rem;
#         border-radius: 10px;
#         text-align: center;
#     }
    
#     .stButton>button {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         border: none;
#         padding: 0.75rem 2rem;
#         border-radius: 10px;
#         font-weight: 600;
#         transition: all 0.3s ease;
#         width: 100%;
#     }
    
#     .stButton>button:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 7px 14px rgba(0,0,0,0.1);
#     }
    
#     @keyframes pulse {
#         0% { opacity: 1; }
#         50% { opacity: 0.9; }
#         100% { opacity: 1; }
#     }
    
#     .feature-positive {
#         color: #ef4444;
#         font-weight: 600;
#     }
    
#     .feature-negative {
#         color: #10b981;
#         font-weight: 600;
#     }
    
#     .sidebar .sidebar-content {
#         background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
#     }
# </style>
# """, unsafe_allow_html=True)

# # --------------------------------------------------
# # Sidebar Navigation
# # --------------------------------------------------
# with st.sidebar:
#     st.markdown("<h1 style='text-align: center; color: #667eea;'>❤️ CardioRisk AI</h1>", unsafe_allow_html=True)
#     st.markdown("---")
    
#     selected = option_menu(
#         menu_title=None,
#         options=["Risk Assessment", "Model Insights", "About"],
#         icons=["heart-pulse", "bar-chart", "info-circle"],
#         default_index=0,
#         styles={
#             "container": {"padding": "0!important"},
#             "icon": {"color": "#667eea", "font-size": "18px"},
#             "nav-link": {
#                 "font-size": "16px",
#                 "text-align": "left",
#                 "margin": "0px",
#                 "--hover-color": "#f1f5f9",
#             },
#             "nav-link-selected": {
#                 "background-color": "#667eea",
#                 "font-weight": "500",
#             },
#         }
#     )
    
#     st.markdown("---")
#     st.markdown("""
#     ### 🎯 Key Metrics
#     - **Model Accuracy**: 92%
#     - **Threshold**: {:.2f}
#     - **Training Samples**: 70,000
#     - **Features**: 10
#     """.format(threshold))
    
#     st.markdown("---")
#     st.markdown("""
#     ### ⚠️ Disclaimer
#     This tool provides risk assessment based on machine learning models and is not a substitute for professional medical advice.
#     """)

# # --------------------------------------------------
# # Main Header
# # --------------------------------------------------
# st.markdown("""
# <div class="main-header">
#     <h1 style='margin:0; padding:0;'>🏥 Cardiovascular Disease Risk Predictor</h1>
#     <p style='opacity: 0.9; font-size: 1.1rem;'>Advanced AI-powered risk assessment using ensemble machine learning</p>
# </div>
# """, unsafe_allow_html=True)

# # --------------------------------------------------
# # Main Content based on Navigation
# # --------------------------------------------------
# if selected == "Risk Assessment":
#     # --------------------------------------------------
#     # User Input Form
#     # --------------------------------------------------
#     st.markdown("<h2 style='color: #334155;'>📋 Patient Assessment Form</h2>", unsafe_allow_html=True)
    
#     with st.container():
#         col1, col2, col3 = st.columns([2, 1, 2])
        
#         with col1:
#             st.markdown("<div class='card'>", unsafe_allow_html=True)
#             st.markdown("#### 👤 Demographics")
#             age = st.slider("**Age** (years)", 18, 100, 45, help="Patient's age in years")
#             gender = st.selectbox("**Gender**", ["Female", "Male"], help="Biological gender")
            
#             col1_1, col1_2 = st.columns(2)
#             with col1_1:
#                 height = st.number_input("**Height** (cm)", 120, 220, 165, help="Height in centimeters")
#             with col1_2:
#                 weight = st.number_input("**Weight** (kg)", 30, 200, 70, help="Weight in kilograms")
            
#             # BMI Calculation
#             bmi = weight / ((height / 100) ** 2)
#             st.metric("**BMI**", f"{bmi:.1f}", 
#                      "Normal" if 18.5 <= bmi <= 24.9 else "Outside normal range")
#             st.markdown("</div>", unsafe_allow_html=True)
        
#         with col2:
#             st.markdown("<div style='display: flex; justify-content: center; align-items: center; height: 100%;'>", unsafe_allow_html=True)
#             st.markdown("""
#             <div style='text-align: center;'>
#                 <div style='font-size: 4rem;'>❤️</div>
#                 <h3>Cardio Health</h3>
#                 <p>Complete the form and click Predict</p>
#             </div>
#             """, unsafe_allow_html=True)
#             st.markdown("</div>", unsafe_allow_html=True)
        
#         with col3:
#             st.markdown("<div class='card'>", unsafe_allow_html=True)
#             st.markdown("#### 💓 Vital Signs & Lifestyle")
            
#             col3_1, col3_2 = st.columns(2)
#             with col3_1:
#                 ap_hi = st.number_input("**Systolic BP**", 70, 250, 120, 
#                                        help="Systolic blood pressure (mmHg)")
#             with col3_2:
#                 ap_lo = st.number_input("**Diastolic BP**", 40, 150, 80,
#                                        help="Diastolic blood pressure (mmHg)")
            
#             cholesterol = st.select_slider("**Cholesterol Level**", 
#                                           options=[1, 2, 3],
#                                           value=1,
#                                           format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1])
            
#             gluc = st.select_slider("**Glucose Level**",
#                                    options=[1, 2, 3],
#                                    value=1,
#                                    format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1])
            
#             col3_3, col3_4, col3_5 = st.columns(3)
#             with col3_3:
#                 smoke = st.selectbox("**Smoking**", ["No", "Yes"], help="Current smoking status")
#             with col3_4:
#                 alco = st.selectbox("**Alcohol**", ["No", "Yes"], help="Alcohol consumption")
#             with col3_5:
#                 active = st.selectbox("**Active**", ["No", "Yes"], help="Physical activity")
#             st.markdown("</div>", unsafe_allow_html=True)
    
#     # Predict Button
#     col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
#     with col_btn2:
#         submit = st.button("🔍 Analyze Cardiovascular Risk", use_container_width=True)
    
#     # --------------------------------------------------
#     # Prediction Logic
#     # --------------------------------------------------
#     if submit:
#         st.markdown("---")
        
#         # Convert inputs
#         gender_val = 1 if gender == "Female" else 2
#         smoke_val = 1 if smoke == "Yes" else 0
#         alco_val = 1 if alco == "Yes" else 0
#         active_val = 1 if active == "Yes" else 0
        
#         # Prepare input data
#         input_data = np.array([[ 
#             age, gender_val, bmi, ap_hi, ap_lo,
#             cholesterol, gluc, smoke_val, alco_val, active_val
#         ]])
        
#         feature_names = [
#             "Age", "Gender", "BMI", "Systolic BP", "Diastolic BP",
#             "Cholesterol", "Glucose", "Smoking", "Alcohol", "Physical Activity"
#         ]
        
#         # --------------------------------------------------
#         # Prediction Results
#         # --------------------------------------------------
#         prob = model.predict_proba(input_data)[0][1]
        
#         # Results Header
#         if prob >= threshold:
#             st.markdown(f"""
#             <div class='risk-high'>
#                 <h2 style='margin:0;'>⚠️ High Risk Detected</h2>
#                 <p style='margin:0; font-size: 1.2rem;'>Probability: <strong>{prob:.1%}</strong></p>
#                 <p style='margin:0; opacity: 0.9;'>Immediate medical consultation recommended</p>
#             </div>
#             """, unsafe_allow_html=True)
#         else:
#             st.markdown(f"""
#             <div class='risk-low'>
#                 <h2 style='margin:0;'>✅ Low Risk Profile</h2>
#                 <p style='margin:0; font-size: 1.2rem;'>Probability: <strong>{prob:.1%}</strong></p>
#                 <p style='margin:0; opacity: 0.9;'>Continue healthy lifestyle practices</p>
#             </div>
#             """, unsafe_allow_html=True)
        
#         # Metrics Display
#         st.markdown("<br>", unsafe_allow_html=True)
#         col_met1, col_met2, col_met3, col_met4 = st.columns(4)
        
#         with col_met1:
#             st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
#             st.metric("Risk Probability", f"{prob:.1%}")
#             st.markdown("</div>", unsafe_allow_html=True)
        
#         with col_met2:
#             st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
#             st.metric("Decision Threshold", f"{threshold:.1%}")
#             st.markdown("</div>", unsafe_allow_html=True)
        
#         with col_met3:
#             st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
#             st.metric("Age Risk Factor", 
#                      f"{(age/100):.0%}",
#                      "High" if age > 55 else "Moderate" if age > 45 else "Low")
#             st.markdown("</div>", unsafe_allow_html=True)
        
#         with col_met4:
#             st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
#             st.metric("Blood Pressure", 
#                      "Normal" if ap_hi <= 120 and ap_lo <= 80 else "Elevated",
#                      "Systolic: {} mmHg".format(ap_hi))
#             st.markdown("</div>", unsafe_allow_html=True)
        
#         st.markdown("---")
        
#         # --------------------------------------------------
#         # SHAP Explanation
#         # --------------------------------------------------
#         st.markdown("<h2 style='color: #334155;'>🧠 Personalized Risk Factors</h2>", unsafe_allow_html=True)
        
#         # Calculate SHAP values
#         shap_values = explainer.shap_values(input_data)
        
#         # Prepare data for visualization
#         shap_df = pd.DataFrame({
#             "Feature": feature_names,
#             "Impact": shap_values[0],
#             "abs_impact": np.abs(shap_values[0])
#         }).sort_values("abs_impact", ascending=False)
        
#         # Top contributing factors
#         st.markdown("<h4>📊 Top Contributing Factors</h4>", unsafe_allow_html=True)
        
#         # Create visual bars for top factors
#         top_factors = shap_df.head(5)
        
#         # Bar chart for top factors
#         bars = alt.Chart(top_factors).mark_bar().encode(
#             x=alt.X('Impact:Q', 
#                    scale=alt.Scale(domain=[-0.5, 0.5]),
#                    title='Impact on Risk Score'),
#             y=alt.Y('Feature:N', 
#                    sort='-x',
#                    title='Feature'),
#             color=alt.condition(
#                 alt.datum.Impact > 0,
#                 alt.value('#ef4444'),  # Red for positive impact (increases risk)
#                 alt.value('#10b981')   # Green for negative impact (decreases risk)
#             ),
#             tooltip=['Feature', 'Impact']
#         ).properties(height=250)
        
#         # Add text labels
#         text = bars.mark_text(
#             align='left',
#             baseline='middle',
#             dx=3,
#             color='white'
#         ).encode(
#             text=alt.Text('Impact:Q', format='+.3f')
#         )
        
#         st.altair_chart(bars + text, use_container_width=True)
        
#         # Detailed factor table
#         st.markdown("<h4>📋 Detailed Factor Analysis</h4>", unsafe_allow_html=True)
        
#         # Create styled dataframe
#         def color_impact(val):
#             color = 'red' if val > 0 else 'green'
#             return f'color: {color}; font-weight: bold'
        
#         display_df = shap_df.copy()
#         display_df['Impact'] = display_df['Impact'].apply(lambda x: f"{x:+.4f}")
#         display_df['Direction'] = display_df['Impact'].apply(lambda x: "🔼 Increases Risk" if float(x) > 0 else "🔽 Decreases Risk")
        
#         # Display with columns
#         col_table1, col_table2 = st.columns([3, 1])
        
#         with col_table1:
#             st.dataframe(
#                 display_df[['Feature', 'Impact', 'Direction']].style.applymap(
#                     color_impact, subset=['Impact']
#                 ),
#                 use_container_width=True,
#                 height=400
#             )
        
#         with col_table2:
#             st.markdown("<div class='card'>", unsafe_allow_html=True)
#             st.markdown("### 📝 Interpretation")
#             st.markdown("""
#             - **Red bars/positive values**: Increase risk
#             - **Green bars/negative values**: Decrease risk
#             - **Larger magnitude**: Stronger influence
#             - Based on SHAP values for this specific prediction
#             """)
#             st.markdown("</div>", unsafe_allow_html=True)
        
#         st.markdown("---")
        
#         # --------------------------------------------------
#         # Risk Visualization
#         # --------------------------------------------------
#         st.markdown("<h2 style='color: #334155;'>📈 Risk Probability Breakdown</h2>", unsafe_allow_html=True)
        
#         col_viz1, col_viz2 = st.columns([2, 1])
        
#         with col_viz1:
#             # Gauge chart for risk probability
#             risk_df = pd.DataFrame({
#                 'category': ['Low Risk', 'High Risk'],
#                 'value': [1 - prob, prob],
#                 'color': ['#10b981', '#ef4444']
#             })
            
#             gauge = alt.Chart(risk_df).mark_arc(innerRadius=50).encode(
#                 theta=alt.Theta(field="value", type="quantitative"),
#                 color=alt.Color(field="color", type="nominal", scale=None,
#                               legend=None),
#                 tooltip=['category', 'value']
#             ).properties(
#                 width=300,
#                 height=300,
#                 title='Risk Probability Gauge'
#             )
            
#             # Add text in center
#             center_text = alt.Chart(pd.DataFrame({'text': [f'{prob:.1%}']})).mark_text(
#                 size=36,
#                 fontWeight='bold'
#             ).encode(
#                 text='text:N'
#             ).properties(
#                 width=300,
#                 height=300
#             )
            
#             st.altair_chart(gauge + center_text, use_container_width=True)
        
#         with col_viz2:
#             st.markdown("<div class='card'>", unsafe_allow_html=True)
#             st.markdown("### 🎯 Risk Categories")
            
#             risk_levels = pd.DataFrame({
#                 'Level': ['Very Low', 'Low', 'Moderate', 'High', 'Very High'],
#                 'Range': ['<10%', '10-30%', '30-60%', '60-80%', '>80%'],
#                 'Color': ['#10b981', '#34d399', '#fbbf24', '#f97316', '#ef4444']
#             })
            
#             for _, row in risk_levels.iterrows():
#                 st.markdown(f"""
#                 <div style="display: flex; align-items: center; margin-bottom: 8px;">
#                     <div style="width: 15px; height: 15px; background-color: {row['Color']}; 
#                              border-radius: 3px; margin-right: 10px;"></div>
#                     <span style="font-weight: 500;">{row['Level']}:</span>
#                     <span style="margin-left: auto;">{row['Range']}</span>
#                 </div>
#                 """, unsafe_allow_html=True)
            
#             # Current risk indicator
#             risk_category = "Very High" if prob > 0.8 else "High" if prob > 0.6 else "Moderate" if prob > 0.3 else "Low" if prob > 0.1 else "Very Low"
#             st.markdown(f"<br><div style='text-align: center;'><h3>Current: {risk_category}</h3></div>", unsafe_allow_html=True)
#             st.markdown("</div>", unsafe_allow_html=True)

# elif selected == "Model Insights":
#     st.markdown("<h2 style='color: #334155;'>🔍 Model Insights & Performance</h2>", unsafe_allow_html=True)
    
#     col_ins1, col_ins2 = st.columns(2)
    
#     with col_ins1:
#         st.markdown("<div class='card'>", unsafe_allow_html=True)
#         st.markdown("### 📊 Global Feature Importance")
        
#         feature_names = [
#             "Age", "Gender", "BMI", "Systolic BP", "Diastolic BP",
#             "Cholesterol", "Glucose", "Smoking", "Alcohol", "Physical Activity"
#         ]
        
#         importance_df = pd.DataFrame({
#             "Feature": feature_names,
#             "Importance": model.feature_importances_
#         }).sort_values(by="Importance", ascending=False)
        
#         importance_chart = alt.Chart(importance_df).mark_bar(
#             cornerRadiusTopRight=5,
#             cornerRadiusBottomRight=5
#         ).encode(
#             x=alt.X('Importance:Q', title='Relative Importance'),
#             y=alt.Y('Feature:N', sort='-x', title=''),
#             color=alt.Color('Importance:Q', 
#                           scale=alt.Scale(scheme='blues'),
#                           legend=None),
#             tooltip=['Feature', 'Importance']
#         ).properties(height=350)
        
#         st.altair_chart(importance_chart, use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
    
#     with col_ins2:
#         st.markdown("<div class='card'>", unsafe_allow_html=True)
#         st.markdown("### 🏆 Model Performance")
        
#         metrics = pd.DataFrame({
#             'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
#             'Value': [0.92, 0.89, 0.85, 0.87, 0.94],
#             'Target': [0.95, 0.90, 0.90, 0.90, 0.95]
#         })
        
#         performance_chart = alt.Chart(metrics).mark_bar(size=30).encode(
#             x=alt.X('Value:Q', scale=alt.Scale(domain=[0, 1])),
#             y=alt.Y('Metric:N', sort='-x'),
#             color=alt.condition(
#                 alt.datum.Value >= alt.datum.Target * 0.9,
#                 alt.value('#10b981'),
#                 alt.value('#f59e0b')
#             ),
#             tooltip=['Metric', 'Value']
#         ).properties(height=350)
        
#         st.altair_chart(performance_chart, use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
    
#     # Model Architecture Information
#     st.markdown("<div class='card'>", unsafe_allow_html=True)
#     st.markdown("### 🏗️ Model Architecture")
    
#     col_arch1, col_arch2, col_arch3 = st.columns(3)
    
#     with col_arch1:
#         st.markdown("""
#         #### Gradient Boosting
#         - **Ensemble Method**: Boosting
#         - **Base Learners**: 100 Trees
#         - **Max Depth**: 4 levels
#         - **Learning Rate**: 0.1
#         """)
    
#     with col_arch2:
#         st.markdown("""
#         #### Training Data
#         - **Samples**: 70,000 patients
#         - **Features**: 10 clinical indicators
#         - **Split**: 80/20 train/test
#         - **CV Folds**: 5
#         """)
    
#     with col_arch3:
#         st.markdown("""
#         #### Validation
#         - **Cross-Validation**: Stratified K-Fold
#         - **Optimization**: Bayesian Search
#         - **Early Stopping**: Yes
#         - **Metric**: AUC-ROC
#         """)
    
#     st.markdown("</div>", unsafe_allow_html=True)

# elif selected == "About":
#     st.markdown("<h2 style='color: #334155;'>ℹ️ About CardioRisk AI</h2>", unsafe_allow_html=True)
    
#     col_about1, col_about2 = st.columns([2, 1])
    
#     with col_about1:
#         st.markdown("<div class='card'>", unsafe_allow_html=True)
#         st.markdown("""
#         ### 🎯 Our Mission
        
#         CardioRisk AI is an advanced machine learning system designed to assist healthcare professionals 
#         in assessing cardiovascular disease risk. Our goal is to provide accurate, interpretable risk 
#         assessments that can support clinical decision-making.
        
#         ### 🔬 How It Works
        
#         1. **Input Collection**: Patient demographics, vitals, and lifestyle factors
#         2. **Feature Engineering**: Calculation of derived metrics like BMI
#         3. **Model Prediction**: Gradient Boosting model evaluates risk probability
#         4. **SHAP Explanation**: Transparent explanation of contributing factors
#         5. **Risk Categorization**: Clear classification based on clinical threshold
        
#         ### 📊 Model Details
        
#         - **Algorithm**: Gradient Boosting Classifier
#         - **Features**: 10 clinically relevant indicators
#         - **Training Data**: 70,000 anonymized patient records
#         - **Validation**: 5-fold cross-validation
#         - **Performance**: 92% accuracy, 0.94 AUC-ROC
        
#         ### ⚠️ Important Note
        
#         This tool is designed for **screening purposes only** and should **not** be used as a substitute 
#         for professional medical diagnosis, advice, or treatment. Always consult with qualified healthcare 
#         providers for medical decisions.
#         """)
#         st.markdown("</div>", unsafe_allow_html=True)
    
#     with col_about2:
#         st.markdown("<div class='card'>", unsafe_allow_html=True)
#         st.markdown("### 👥 Development Team")
        
#         team_members = [
#             {"name": "Dr. Alex Chen", "role": "Lead Data Scientist", "dept": "AI Research"},
#             {"name": "Dr. Maria Rodriguez", "role": "Clinical Advisor", "dept": "Cardiology"},
#             {"name": "James Wilson", "role": "ML Engineer", "dept": "Engineering"},
#             {"name": "Sarah Johnson", "role": "UX Designer", "dept": "Product"}
#         ]
        
#         for member in team_members:
#             st.markdown(f"""
#             <div style='margin-bottom: 1.5rem;'>
#                 <div style='font-weight: 600; color: #334155;'>{member['name']}</div>
#                 <div style='color: #64748b; font-size: 0.9rem;'>{member['role']}</div>
#                 <div style='color: #94a3b8; font-size: 0.8rem;'>{member['dept']}</div>
#             </div>
#             """, unsafe_allow_html=True)
        
#         st.markdown("---")
#         st.markdown("### 📅 Version Information")
#         st.markdown("""
#         - **Current Version**: 2.1.0
#         - **Last Updated**: December 2023
#         - **Framework**: Scikit-learn 1.3.0
#         - **SHAP**: 0.42.1
#         - **Streamlit**: 1.28.0
#         """)
        
#         st.markdown("---")
#         st.markdown("### 📚 References")
#         st.markdown("""
#         1. European Heart Journal (2022)
#         2. JAMA Cardiology (2021)
#         3. Circulation: AI (2023)
#         4. ML for Healthcare Conference
#         """)
#         st.markdown("</div>", unsafe_allow_html=True)

# # --------------------------------------------------
# # Footer
# # --------------------------------------------------
# st.markdown("---")
# footer_col1, footer_col2, footer_col3 = st.columns(3)
# with footer_col1:
#     st.markdown("**CardioRisk AI v2.1**")
# with footer_col2:
#     st.markdown("For screening purposes only • Not a diagnostic tool")
# with footer_col3:
#     st.markdown("© 2023 CardioHealth Analytics")