import streamlit as st
import pickle
import pandas as pd

# Set up page configurations
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")

st.title("🎓 Student Final Score Predictor")
st.write("Fill out the details below to estimate the student's final score.")
st.markdown("---")

# 1. Load your saved model and label encoder
@st.cache_resource
def load_assets():
    with open('student_score_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return model, encoder

try:
    model, encoder = load_assets()
except FileNotFoundError:
    st.error("⚠️ Error: 'student_score_model.pkl' or 'label_encoder.pkl' not found. Please ensure they are in the same folder as this script.")
    st.stop()

# 2. Create the layout and UI inputs (Omitting Age input)
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", options=["Male", "Female"])
    study_hours = st.slider("Study Hours per Week", min_value=0, max_value=100, value=15, step=1)
    attendance = st.slider("Attendance Rate (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
    # Automatically grab the options from your label encoder classes
    # Manually providing the education levels from your dataset
    parent_edu = st.selectbox("Parent Education Level", options=["Bachelor", "High School", "Master", "PhD", "None"])

with col2:
    internet = st.selectbox("Has Internet Access?", options=["Yes", "No"])
    extracurricular = st.selectbox("Participates in Extracurriculars?", options=["Yes", "No"])
    prev_score = st.slider("Previous Test Score", min_value=0, max_value=100, value=70, step=1)

# 3. Handle data mapping & hardcoded constraints
# Hardcode age to 18 for every prediction behind the scenes
age = 18 

gender_encoded = 1 if gender == "Male" else 0
internet_encoded = 1 if internet == "Yes" else 0
extra_encoded = 1 if extracurricular == "Yes" else 0
parent_edu_map = {"Bachelor": 0, "High School": 1, "Master": 2, "PhD": 3, "None": 4}
parent_edu_encoded = parent_edu_map[parent_edu]

# 4. Predict Button and Inference
st.markdown("---")
if st.button("Predict Final Score", type="primary", use_container_width=True):
    
    # Create the input row ensuring the column names match your training dataframe
    input_row = {
        'gender': gender_encoded,
        'age': age,
        'study_hours_per_week': study_hours,
        'attendance_rate': attendance,
        'parent_education': parent_edu_encoded,
        'internet_access': internet_encoded,
        'extracurricular': extra_encoded,
        'previous_score': prev_score
    }
    
    # Transform to DataFrame
    df_input = pd.DataFrame([input_row])
    
    # Strictly enforce the exact column sequence your linear regression model expects
    column_order = [
        'gender', 'age', 'study_hours_per_week', 'attendance_rate', 
        'parent_education', 'internet_access', 'extracurricular', 'previous_score'
    ]
    df_input = df_input[column_order]
    
    # Generate prediction
    predicted_score = model.predict(df_input)[0]
    
    # Clamp score between 0 and 100 just in case linear regression overshoots
    final_output = max(0.0, min(100.0, predicted_score))
    
    # Display Result
    st.success(f"### 🎯 Predicted Final Score: **{final_output:.2f}**")