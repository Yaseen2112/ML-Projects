import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Responsive Page Setup and Theme Colors ---
st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="wide"
)

# --- Custom CSS for Modern, Clean UI, Animations & Accessibility ---
st.markdown("""
<style>
  .app-title {
    text-align: center;
    font-size: clamp(2.2rem, 6vw, 3.2rem);
    font-weight: 900;
    color: #c02433;
    margin-bottom: 10px;
    letter-spacing: 2px;
    text-shadow: 0 2px 8px #fff2ee, 0 0px 0px #c02433;
    font-family: 'Segoe UI', Arial, sans-serif;
    background: linear-gradient(90deg, #c02433 0%, #7d2131 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .app-subtitle {
    text-align: center;
    font-size: clamp(1.1rem, 3vw, 1.5rem);
    color: #7d2131;
    margin-bottom: 30px;
    font-family: 'Segoe UI', Arial, sans-serif;
  }
  .caption-custom {
    text-align: center;
    font-size: 1.2rem;
    color: #555555;
    margin-top: 3rem;
    line-height: 1.8;
    font-family: 'Arial', sans-serif;
    font-weight: 700;
    letter-spacing: 0.03em;
  }
  div.stForm button[kind="primary"] {
    background: linear-gradient(90deg, #c02433 0%, #7d2131 100%) !important;
    color: white !important;
    font-weight: 900 !important;
    font-size: 1.2rem !important;
    border-radius: 10px !important;
    border: none !important;
    box-shadow: 0 2px 12px #c0243350;
  }
</style>
""", unsafe_allow_html=True)

# --- Animated and Centered Hero Section ---
st.markdown("""
<div class="app-title">HEART DISEASE PREDICTION</div>
<div class="app-subtitle">A modern pulse-themed medical prediction tool</div>
<div style='text-align:center;'>
  <img src='https://img.icons8.com/color/96/heart-with-pulse--v2.png'
       width='72' style='margin-bottom:18px; animation: heartBeat 1.5s infinite; filter: drop-shadow(0 2px 15px #d7263d90);'/>
</div>
""", unsafe_allow_html=True)

# --- Dropdown options for user inputs with friendly strings ---
dropdown_options = {
    "Age": [str(i) for i in range(29, 81)],
    "Sex": ["Female", "Male"],
    "Chest Pain Type": ["Typical angina", "Atypical angina", "Non-anginal", "Asymptomatic"],
    "Resting Blood Pressure": [str(i) for i in range(80, 201)],
    "Cholesterol": [str(i) for i in range(126, 565)],
    "Fasting Blood Sugar": ["No", "Yes"],
    "Resting ECG": ["Normal", "ST-T abnormality", "Left ventricular hypertrophy"],
    "Max Heart Rate": [str(i) for i in range(71, 203)],
    "Exercise Induced Angina": ["No", "Yes"],
    "ST Depression": [f"{x/10:.1f}" for x in range(0, 63)],
    "Slope": ["Upsloping", "Flat", "Downsloping"],
    "No. of Major Vessels": [str(i) for i in range(0, 4)],
    "Thalassemia": ["Normal", "Fixed defect", "Reversible defect"],
}

mapping_dict = {
    "Sex": {"Female": 0, "Male": 1},
    "Chest Pain Type": {"Typical angina": 0, "Atypical angina": 1, "Non-anginal": 2, "Asymptomatic": 3},
    "Fasting Blood Sugar": {"No": 0, "Yes": 1},
    "Resting ECG": {"Normal": 0, "ST-T abnormality": 1, "Left ventricular hypertrophy": 2},
    "Exercise Induced Angina": {"No": 0, "Yes": 1},
    "Slope": {"Upsloping": 0, "Flat": 1, "Downsloping": 2},
    "Thalassemia": {"Normal": 1, "Fixed defect": 2, "Reversible defect": 3},
}

features_left = [
    "Age",
    "Sex",
    "Chest Pain Type",
    "Resting Blood Pressure",
    "Cholesterol",
    "Fasting Blood Sugar",
    "Resting ECG",
]
features_right = [
    "Max Heart Rate",
    "Exercise Induced Angina",
    "ST Depression",
    "Slope",
    "No. of Major Vessels",
    "Thalassemia",
]

def convert_selection(feature, selection):
    if feature in mapping_dict:
        return mapping_dict[feature][selection]
    else:
        try:
            return float(selection) if '.' in selection else int(selection)
        except:
            return 0

# --- Save Patient Details Function ---
def save_patient_details(data_dict, filename="outputs/patient_details.xlsx"):
    # Ensure Name and Contact are the first columns
    ordered_keys = ['Name', 'Contact'] + [k for k in data_dict.keys() if k not in ['Name', 'Contact']]
    df = pd.DataFrame([[data_dict.get(k, "") for k in ordered_keys]], columns=ordered_keys)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if os.path.exists(filename):
        df_existing = pd.read_excel(filename)
        # Reorder columns if needed
        for col in ordered_keys:
            if col not in df_existing.columns:
                df_existing[col] = ""
        df_existing = df_existing[ordered_keys]
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_excel(filename, index=False)

# --- Form for user input ---
if "show_result" not in st.session_state:
    st.session_state.show_result = False
if "inputs" not in st.session_state:
    st.session_state.inputs = {k: dropdown_options[k][0] for k in dropdown_options}

with st.form("input_form", clear_on_submit=False):
    st.markdown("#### Please enter your details below", unsafe_allow_html=True)
    name = st.text_input("Name", key="name_field")
    contact = st.text_input("Contact", key="contact_field")
    col_left, col_right = st.columns([1, 1])
    user_input = []
    changed = False
    with col_left:
        for feature in features_left:
            val = st.selectbox(
                feature, dropdown_options[feature],
                key=f"left_{feature}",
                index=dropdown_options[feature].index(st.session_state.inputs.get(feature, dropdown_options[feature][0]))
            )
            if val != st.session_state.inputs.get(feature, dropdown_options[feature][0]):
                changed = True
            user_input.append((feature, val))
    with col_right:
        for feature in features_right:
            val = st.selectbox(
                feature, dropdown_options[feature],
                key=f"right_{feature}",
                index=dropdown_options[feature].index(st.session_state.inputs.get(feature, dropdown_options[feature][0]))
            )
            if val != st.session_state.inputs.get(feature, dropdown_options[feature][0]):
                changed = True
            user_input.append((feature, val))
    predict_btn = st.form_submit_button(
        "Predict",
        help="Click to get prediction",
        use_container_width=True
    )
    save_btn = st.form_submit_button(
        "Save Details",
        help="Save entered details to Excel",
        use_container_width=True
    )

if changed:
    st.components.v1.html("""
        <script>
        var msg = window.speechSynthesis;
        var utter = new SpeechSynthesisUtterance("Input changed");
        utter.rate = 1.0;
        utter.pitch = 1.0;
        utter.volume = 1.0;
        if (msg.speaking) msg.cancel();
        msg.speak(utter);
        </script>
    """, height=0)

if predict_btn:
    features_names = [f[0] for f in user_input]
    features_vals = [convert_selection(k, v) for k, v in user_input]
    X_new = np.array(features_vals).reshape(1, -1)
    scaler = joblib.load('models/scaler.pkl')
    model = joblib.load('models/rf_heart_model.pkl')
    X_scaled = scaler.transform(X_new)
    pred = model.predict(X_scaled)[0]
    st.session_state.result = pred
    st.session_state.show_result = True
    st.session_state.inputs = {k: v for k, v in user_input}
    st.session_state.name = name
    st.session_state.contact = contact
    result_text = "Heart Disease Detected" if pred == 1 else "No Heart Disease Detected"
    st.components.v1.html(f"""
        <script>
        var msg = window.speechSynthesis;
        var utter = new SpeechSynthesisUtterance("{result_text}");
        utter.rate = 0.9;
        utter.pitch = 1.0;
        utter.volume = 1.0;
        if (msg.speaking) msg.cancel();
        msg.speak(utter);
        </script>
    """, height=0)

if save_btn:
    patient_dict = {k: v for k, v in user_input}
    patient_dict["Name"] = name
    patient_dict["Contact"] = contact
    save_patient_details(patient_dict)
    st.success("Patient details saved to Excel file.")
    st.components.v1.html("""
        <script>
        var msg = window.speechSynthesis;
        var utter = new SpeechSynthesisUtterance("Details saved");
        utter.rate = 1.0;
        utter.pitch = 1.0;
        utter.volume = 1.0;
        if (msg.speaking) msg.cancel();
        msg.speak(utter);
        </script>
    """, height=0)

# --- Show Results ---
if st.session_state.show_result:
    color = "red" if st.session_state.result == 1 else "green"
    icon = "❤" if st.session_state.result == 1 else "✅"
    st.markdown(f"""
    <div class="prediction-card {color}" style="text-align:center;">
        <h2 style="color:{color}; text-align:center;">{icon} {('Heart Disease Detected' if st.session_state.result == 1 else 'No Heart Disease Detected')}</h2>
        <p style="color:#374151; text-align:center;">Based on your input, the model predicts <b>{('presence' if st.session_state.result == 1 else 'absence')}</b> of heart disease.</p>
        <p style="color:#7d2131; text-align:center;"><b>Name:</b> {st.session_state.name if 'name' in st.session_state else ''} &nbsp; <b>Contact:</b> {st.session_state.contact if 'contact' in st.session_state else ''}</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Dynamic Project Info Table ---
    feature_names = list(dropdown_options.keys())
    user_inputs_str = [st.session_state.inputs.get(f, "") for f in feature_names]
    table_data = []
    for attr, val_str in zip(feature_names, user_inputs_str):
        table_data.append({
            "Attribute": attr,
            "Your Value": val_str,
            "Project Info": f"{attr} is a key parameter for heart disease prediction."
        })
    df_table = pd.DataFrame(table_data)

    def info_color(row):
        return ['color: black; font-weight: bold;' if col == 'Project Info' else '' for col in row.index]

    st.markdown("### Your Input Summary and Project Insights")
    st.write(
        df_table.style.apply(info_color, axis=1)
        .set_properties(**{'text-align': 'center', 'font-family': 'Arial, sans-serif'})
        .set_table_styles([{'selector': 'th', 'props': [('background-color', '#c02433'), ('color', 'white'), ('font-weight', 'bold')]}])
    )

# --- Feature Importance Visual (Impact) ---
st.markdown("## Feature Impact on Prediction")
model = joblib.load('models/rf_heart_model.pkl')
features = list(dropdown_options.keys())
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(7, 4))
bar_colors = plt.cm.RdPu(np.linspace(0.5, 1, len(indices)))
ax.barh([features[i] for i in indices], importances[indices], color=bar_colors, edgecolor='#7d2131', height=0.55)
ax.set_xlabel('Importance', fontsize=11, color='#7d2131')
ax.set_title('Feature Impact on Heart Disease Prediction', fontsize=13, color='#c02433', pad=12)
ax.invert_yaxis()
ax.grid(axis='x', linestyle='--', alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

# --- About Section ---
st.markdown("""
<div style='max-width: 900px; margin: 0 auto; text-align: center; font-family: Arial, sans-serif; color: #4a4a4a;'>
  <h2 style='color: #c02433; font-weight: 900; letter-spacing: 1px;'>About Heart Disease Prediction</h2>
  <p style='font-size: 1.1rem; line-height: 1.6; margin-top: 0.5rem;'>
    This predictive tool uses state-of-the-art machine learning techniques to analyze key clinical parameters and flag the potential presence of heart disease.
    By inputting your medical and lifestyle data, you receive a personalized assessment backed by validated models to support early intervention and informed health decisions.
  </p>
  <hr style='width: 60px; border: 2px solid #c02433; margin: 1rem auto 2rem auto;'>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='dashed'>", unsafe_allow_html=True)
st.markdown("""
<div class="caption-custom" style="text-align:center;">
Created with ❤ & Streamlit | Pulse Medical Theme | Modern UI, accessibility, and audio feedback &copy; 2025<br>
SHAIK YASEEN &nbsp;|&nbsp; 22BQ1A05K4@vvit.net<br>
</div>
""", unsafe_allow_html=True)

