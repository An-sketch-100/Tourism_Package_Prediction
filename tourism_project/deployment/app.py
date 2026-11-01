import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configure Streamlit page layout and theme
st.set_page_config(
    page_title="Wellness Tourism Predictor",
    page_icon="✈️",
    layout="centered"
)

# Safely load artifacts from the deployment directory relative to this script
@st.cache_resource
def load_artifacts():
    current_dir = os.path.dirname(__file__)
    preprocessor_path = os.path.join(current_dir, "preprocessor.joblib")
    model_path = os.path.join(current_dir, "best_model.joblib")
    
    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    return preprocessor, model

try:
    preprocessor, model = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    st.error(f"Error loading serialized assets: {e}")
    artifacts_loaded = False

# App Headers
st.title("✈️ Wellness Tourism Package Predictor")
st.markdown("""
This predictive tool uses our production **XGBoost Classifier** to evaluate whether a client is likely to purchase 
the newly introduced **Wellness Tourism Package** before a sales representative initiates contact. 
Complete the customer profile below to run a real-time estimation.
""")

if artifacts_loaded:
    # Create organized input categories using tabs
    tab1, tab2 = st.tabs(["👤 Demographics & Profile", "📞 Sales Engagement"])
    
    with tab1:
        st.subheader("Customer Information")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
            gender = st.selectbox("Gender", ["Female", "Male"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
            designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
            
        with col2:
            monthly_income = st.number_input("Monthly Income (INR)", min_value=1000, max_value=200000, value=25000, step=500)
            city_tier = st.selectbox("City Tier (1 = Highest)", [1, 2, 3], index=0)
            passport = st.selectbox("Has Valid Passport?", ["Yes", "No"], index=1)
            own_car = st.selectbox("Owns a Car?", ["Yes", "No"], index=1)
            
    with tab2:
        st.subheader("Interaction Metrics")
        col3, col4 = st.columns(2)
        
        with col3:
            typeof_contact = st.selectbox("Type of Contact", ["Self Inquiry", "Company Invited"])
            product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
            duration_of_pitch = st.number_input("Duration of Pitch (Minutes)", min_value=1.0, max_value=120.0, value=15.0, step=1.0)
            
        with col4:
            num_person_visiting = st.number_input("Number of Persons Accompanying", min_value=1, max_value=10, value=2, step=1)
            num_children_visiting = st.number_input("Number of Children Accompanying (Under 5)", min_value=0, max_value=5, value=0, step=1)
            num_trips = st.number_input("Average Annual Trips Taken", min_value=1, max_value=30, value=2, step=1)
            preferred_star = st.selectbox("Preferred Hotel Star Rating", [3, 4, 5], index=0)
            num_followups = st.number_input("Total Sales Follow-ups", min_value=1, max_value=10, value=3, step=1)
            pitch_satisfaction = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3, step=1)

    # Assemble the inputs into a dictionary mapping exactly to the dataset schema
    input_dict = {
        'Age': float(age),
        'TypeofContact': typeof_contact,
        'CityTier': int(city_tier),
        'DurationOfPitch': float(duration_of_pitch),
        'Occupation': occupation,
        'Gender': gender,
        'NumberOfPersonVisiting': int(num_person_visiting),
        'NumberOfFollowups': float(num_followups),
        'ProductPitched': product_pitched,
        'PreferredPropertyStar': float(preferred_star),
        'MaritalStatus': marital_status,
        'NumberOfTrips': float(num_trips),
        'Passport': int(1 if passport == "Yes" else 0),
        'PitchSatisfactionScore': int(pitch_satisfaction),
        'OwnCar': int(1 if own_car == "Yes" else 0),
        'NumberOfChildrenVisiting': float(num_children_visiting),
        'Designation': designation,
        'MonthlyIncome': float(monthly_income)
    }

    # Convert the inputs to a single-row Pandas DataFrame
    input_df = pd.DataFrame([input_dict])

    st.markdown("---")
    
    # Trigger prediction on button click
    if st.button("Run Predictor Run", use_container_width=True):
        # 1. Run raw columns through the column transformer preprocessor
        try:
            transformed_features = preprocessor.transform(input_df)
            feature_names = preprocessor.get_feature_names_out()
            X_processed = pd.DataFrame(transformed_features, columns=feature_names)
            
            # 2. Extract probability rankings and final binary labels
            proba = model.predict_proba(X_processed)[0][1]
            prediction = model.predict(X_processed)[0]
            
            # 3. Present intuitive layout feedback based on inference calculations
            st.subheader("Analysis and Prediction")
            
            # Output metric indicators
            m1, m2 = st.columns(2)
            m1.metric(label="Purchase Likelihood Probability", value=f"{proba*100:.2f}%")
            
            if prediction == 1:
                m2.metric(label="System Status Recommendation", value="🎯 TARGET CUSTOMER")
                st.success("**High Purchase Intent Expected!** This customer fits the historical profile of Wellness Tourism Package buyers. We highly recommend proceeding with a personalized premium campaign.")
                st.balloons()
            else:
                m2.metric(label="System Status Recommendation", value="⏸️ LOW PRIORITY")
                st.warning("**Low Purchase Intent Expected.** The predicted behavior indicates they are unlikely to purchase the Wellness Tourism Package at this time. Standard automated follow-ups are advised to save sales resources.")
                
        except Exception as e:
            st.error(f"An error occurred during transformation or inference execution: {e}")
else:
    st.warning("Waiting for production models to be successfully loaded...")
