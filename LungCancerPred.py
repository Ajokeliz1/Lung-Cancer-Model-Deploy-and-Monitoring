import streamlit as st
import pickle
import pandas as pd


try:
    with open('cancer_risk_model.pkl', 'rb') as f:
        best_model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: Model file not found. Please ensure 'cancer_risk_model.pkl' is in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://tse4.mm.bing.net/th/id/OIP.EUo_GMkHi6bt3lKDw-hSOQHaE8?r=0&pid=ImgDet&w=178&h=118&c=7&dpr=1.5");
             background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        position: relative;
    }

    /* Overlay to darken the background */
    .stApp::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        height: 100%;
        width: 100%;
        background-color: rgba(0, 0, 0, 0.01);  /* Adjust 0.01 for stronger/weaker fade */
        z-index: 0;
    }
    
    /* Push content above overlay */
    .stApp > * {
        position: relative;
        z-index: 1;
    
    }
    /* Title (st.title) */
    .stApp h1 {
        color: #FF00FF !important;
        font-size: 40px !important;
        font-weight: bold !important;
    
    }

    /* Header (st.header)) */
    .stApp h2 {
        color: #000000 !important;
        font-size: 35px !important;
    }

     /* Subheader (st.subheader) */
    .stApp h3 {
        color: #FFA500 !important;  /* Orange tone for contrast */
        font-size: 26px !important;
        font-weight: bold !important;
        font-style: italic !important;
    }
    /* Paragraph text */
    .stApp p {
        color: #f0f0f0 !important;
        font-size: 18px !important;
        font-weight: bold !important;
        font-style: italic !important;
    }

    /* General text */
    .stApp p {
        color: #000000 !important;
        font-size: 18px !important;
        font-weight: bold !important;
    }
    /* Push all Streamlit content above the overlay */
    .stApp > * {
        position: relative;
        z-index: 1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.image("lung pic.jpeg", caption="Awareness", use_container_width=True)

st.title(" 🫁 Lung Cancer Risk Prediction App")
st.write("This app predicts the risk level of cancer based on various patient attributes using a trained XGBoost model.")

# Add explanations
st.sidebar.subheader(" About the Model and Prediction")
st.sidebar.write("""
This app uses an XGBoost model, a powerful machine learning algorithm, to predict the likelihood of different cancer risk levels (Low, Medium, High).
XGBoost is known for its efficiency and performance in various classification tasks.
""")

st.sidebar.subheader("Input Features Explained")
st.sidebar.write("""
Please provide the following information to get a cancer risk prediction:
*   **Air Pollution Score (1-8):** Represents the level of exposure to air pollution. Higher values indicate greater exposure.
*   **Alcohol Use Score (1-8):** Represents the frequency and amount of alcohol consumption. Higher values indicate more frequent/higher consumption.
*   **Smoking Score (1-8):** Represents the level of smoking habits (including passive smoking). Higher values indicate more significant smoking exposure.
*   **Genetic Risk Score (1-8):** Represents the inherited predisposition to cancer. Higher values indicate a greater genetic risk.
*   **Chronic Lung Disease Score (1-8):** Represents the presence and severity of chronic lung diseases. Higher values indicate more severe conditions.
*   **Balanced Diet Score (1-8):** Represents the quality and balance of the diet. Higher values indicate a less balanced diet.
*   **Gender:** Biological sex of the patient.
*   **Age Group:** The age category of the patient (Young, Middle, or Senior).

For the score inputs, please provide a value between 1 and 8, where 1 indicates the lowest level/risk and 8 indicates the highest level/risk.
""")

st.header("Enter Patient Data")

# Numerical features with validation
air_pollution = st.number_input("Air Pollution Score (1-8)", min_value=1, max_value=8, value=4)
alcohol_use = st.number_input("Alcohol Use Score (1-8)", min_value=1, max_value=8, value=4)
smoking = st.number_input("Smoking Score (1-8)", min_value=1, max_value=8, value=4)
genetic_risk = st.number_input("Genetic Risk Score (1-8)", min_value=1, max_value=8, value=4)
chronic_lung_disease = st.number_input("Chronic Lung Disease Score (1-8)", min_value=1, max_value=8, value=4)
balanced_diet = st.number_input("Balanced Diet Score (1-8)", min_value=1, max_value=8, value=4)

# Calculate Total_Risk
total_risk = air_pollution + alcohol_use + smoking

# Categorical features
gender = st.radio("Gender", ['Male', 'Female'])
age_group = st.radio("Age Group", ['Young', 'Middle', 'Senior'])

# Process user input
# Create a dictionary to hold the user input values
input_data = {
    'Air Pollution': air_pollution,
    'Alcohol use': alcohol_use,
    'Smoking': smoking,
    'Total_Risk': total_risk,
    'Genetic Risk': genetic_risk,
    'chronic Lung Disease': chronic_lung_disease,
    'Balanced Diet': balanced_diet,
    'Gender_1': False,  # Initialize gender columns
    'Gender_2': False,
    'Age_Group_Young': False, # Initialize age group columns
    'Age_Group_Middle': False,
    'Age_Group_Senior': False,
}

# Map categorical inputs to one-hot encoded columns
if gender == 'Male':
    input_data['Gender_1'] = True
else:
    input_data['Gender_2'] = True

if age_group == 'Young':
    input_data['Age_Group_Young'] = True
elif age_group == 'Middle':
    input_data['Age_Group_Middle'] = True
else:
    input_data['Age_Group_Senior'] = True

# Define the expected features in the correct order as used during training
features = [
    'Air Pollution', 'Alcohol use', 'Smoking', 'Total_Risk',
    'Genetic Risk', 'chronic Lung Disease', 'Balanced Diet',
    'Gender_1', 'Gender_2', 'Age_Group_Young',
    'Age_Group_Middle', 'Age_Group_Senior'
]

# Create a pandas DataFrame from the dictionary, ensuring column order matches X_train
input_df = pd.DataFrame([input_data], columns=features)

st.write("Input Data:")
st.dataframe(input_df)

# Apply bold style to the DataFrame
styled_input_df = input_df.style.map(lambda x: 'font-weight: bold')
st.dataframe(styled_input_df)

# Make predictions with error handling and input validation
if st.button("Predict Risk"):
    # Input validation (basic checks - more comprehensive validation can be added)
    valid_input = True
    for col in ['Air Pollution', 'Alcohol use', 'Smoking', 'Genetic Risk', 'chronic Lung Disease', 'Balanced Diet']:
        if not (1 <= input_df[col].iloc[0] <= 8):
            st.error(f"Invalid input for {col}. Please ensure values are between 1 and 8.")
            valid_input = False
            break

    if valid_input:
        try:
            prediction = best_model.predict(input_df)

            # Display results
            # Map prediction to risk level
            risk_level_map = {0: 'Low', 1: 'Medium', 2: 'High'}
            predicted_risk_level = risk_level_map[prediction[0]]

            # Add explanation of risk levels
            st.subheader("Understanding the Risk Levels")
            st.write("""
            The predicted risk level indicates the likelihood of cancer based on the input features:
            *   **Low Risk:** Indicates a lower probability of having cancer based on the provided data.
            *   **Medium Risk:** Indicates a moderate probability of having cancer based on the provided data.
            *   **High Risk:** Indicates a higher probability of having cancer based on the provided data.

            **Disclaimer:** This prediction is based on a machine learning model trained on a specific dataset and should not be considered a substitute for professional medical advice. Always consult with a qualified healthcare provider for any health concerns.
            """)
            prediction = best_model.predict(input_df)[0]
            probabilities = best_model.predict_proba(input_df)[0]
                
            st.subheader("Prediction Results")
                
            # Display risk level
            risk_level = risk_level_map[prediction]
            if prediction == 2:  # High risk
                st.error(f"🚨 Risk Level: {risk_level} (Probability: {probabilities[prediction]:.1%})")
            elif prediction == 1:  # Medium risk
                st.warning(f"⚠️ Risk Level: {risk_level} (Probability: {probabilities[prediction]:.1%})")
            else:  # Low risk
                st.success(f"✅ Risk Level: {risk_level} (Probability: {probabilities[prediction]:.1%})")
                
            # Show probability distribution
            st.write("Probability Distribution:")
            prob_df = pd.DataFrame({
                'Risk Level': ['Low', 'Medium', 'High'],
                'Probability': probabilities
            })
            st.bar_chart(prob_df.set_index('Risk Level'))
                
             # Show key factors
            st.subheader("Key Contributing Factors")
            feat_importance = pd.Series(best_model.feature_importances_, index=features)
            top_features = feat_importance.nlargest(5)
                
            for feat, importance in top_features.items():
                readable_feat = feat.replace('_', ' ').title()
                st.write(f"- {readable_feat}: {importance:.2f}")

            # Display the prediction
            st.subheader("Predicted Cancer Risk Level:")
            st.write(f"Based on the provided data, the predicted cancer risk level is: **{predicted_risk_level}**")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    

st.sidebar.header("Prediction")
st.sidebar.markdown("""
This app predicts lung cancer risk using:
- Environmental factors (air pollution)
- Lifestyle choices (smoking, alcohol)
- Medical history (genetic risk, lung disease)
""")
st.sidebar.markdown("""
**Risk Level Explanation:**
- Low: 0-30% probability
- Medium: 30-70% probability  
- High: 70-100% probability
""")
