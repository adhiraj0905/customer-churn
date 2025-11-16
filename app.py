import streamlit as st
import pandas as pd
import joblib

# Set the page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="👩",
    layout="wide"
)

# --- LOAD THE MODEL PIPELINE ---
# We load the pipeline we saved in the notebook
model_pipeline = joblib.load('churn_model_pipeline.pkl')


# --- HELPER FUNCTION ---
# This function will take all the inputs and create a dataframe
def make_prediction(gender, SeniorCitizen, Partner, Dependents, tenure,
                    PhoneService, MultipleLines, InternetService, OnlineSecurity,
                    OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
                    StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
                    MonthlyCharges, TotalCharges):

    # Create a dictionary of the inputs
    input_data = {
        'gender': [gender],
        'SeniorCitizen': [1 if SeniorCitizen == 'Yes' else 0], # Convert 'Yes'/'No' to 1/0
        'Partner': [Partner],
        'Dependents': [Dependents],
        'tenure': [tenure],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges]
    }

    # Convert the dictionary to a pandas DataFrame
    # This is crucial because our pipeline expects a DataFrame
    input_df = pd.DataFrame(input_data)

    # Use the loaded pipeline to make a prediction
    # We use 'predict_proba' to get the probability, which is more useful
    # [0][1] selects the probability of the '1' class (Churn)
    prediction_proba = model_pipeline.predict_proba(input_df)[0][1]

    return prediction_proba


# --- STREAMLIT APP LAYOUT ---

st.title("👤 Customer Churn Risk Predictor")
st.markdown("""
This app uses an **XGBoost** model to predict the probability of a customer churning.
Input the customer's details on the left to see the churn risk prediction.
""")

st.divider()

# --- SIDEBAR FOR INPUTS ---
with st.sidebar:
    st.header("Customer Details")
    st.markdown("Enter the customer's information below.")

    # --- Customer Demographics ---
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ("Male", "Female"))
    SeniorCitizen = st.selectbox("Senior Citizen", ("No", "Yes"))
    Partner = st.selectbox("Partner", ("No", "Yes"))
    Dependents = st.selectbox("Dependents", ("No", "Yes"))

    # --- Account Information ---
    st.subheader("Account Information")
    tenure = st.slider("Tenure (months)", 0, 72, 12) # min, max, default
    Contract = st.selectbox("Contract", ("Month-to-month", "One year", "Two year"))
    PaperlessBilling = st.selectbox("Paperless Billing", ("No", "Yes"))
    PaymentMethod = st.selectbox("Payment Method", ("Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"))
    MonthlyCharges = st.slider("Monthly Charges ($)", 0.0, 120.0, 50.0, 0.01)
    TotalCharges = st.slider("Total Charges ($)", 0.0, 9000.0, 1000.0, 0.01)

    # --- Services ---
    st.subheader("Services")
    PhoneService = st.selectbox("Phone Service", ("No", "Yes"))
    MultipleLines = st.selectbox("Multiple Lines", ("No phone service", "No", "Yes"))
    InternetService = st.selectbox("Internet Service", ("DSL", "Fiber optic", "No"))
    OnlineSecurity = st.selectbox("Online Security", ("No internet service", "No", "Yes"))
    OnlineBackup = st.selectbox("Online Backup", ("No internet service", "No", "Yes"))
    DeviceProtection = st.selectbox("Device Protection", ("No internet service", "No", "Yes"))
    TechSupport = st.selectbox("Tech Support", ("No internet service", "No", "Yes"))
    StreamingTV = st.selectbox("Streaming TV", ("No internet service", "No", "Yes"))
    StreamingMovies = st.selectbox("Streaming Movies", ("No internet service", "No", "Yes"))


# --- MAIN PAGE FOR OUTPUT ---

# Create a button to make the prediction
if st.sidebar.button("Predict Churn Risk", use_container_width=True):

    # Call the helper function to get the prediction
    churn_probability = make_prediction(
        gender, SeniorCitizen, Partner, Dependents, tenure,
        PhoneService, MultipleLines, InternetService, OnlineSecurity,
        OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
        StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
        MonthlyCharges, TotalCharges
    )

    # Display the result
    st.subheader("Prediction Result")

    # Format the probability as a percentage
    churn_percent = churn_probability * 100

    # Use columns for a cleaner layout
    col1, col2 = st.columns(2)

    with col1:
        # Display a "metric" card
        st.metric(
            label="Churn Risk Probability",
            value=f"{churn_percent:.2f}%",
            delta=f"{'High Risk' if churn_percent > 60 else 'Medium Risk' if churn_percent > 30 else 'Low Risk'}",
            delta_color="inverse" # 'inverse' makes high deltas red
        )

    with col2:
        # Use a progress bar for a nice visual
        st.write("Risk Level:")
        st.progress(float(churn_probability))

        # Give a business recommendation
        if churn_percent > 60:
            st.error("🔴 **Action:** High priority for retention campaign. Offer discounts or premium support.")
        elif churn_percent > 30:
            st.warning("🟡 **Action:** Medium priority. Monitor and send engagement emails.")
        else:
            st.success("🟢 **Action:** Low risk. No immediate action required.")

else:
    st.info("Please fill in the customer details in the sidebar and click 'Predict Churn Risk'.")