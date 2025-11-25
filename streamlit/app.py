import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. LOAD THE SAVED MODEL AND PREPROCESSOR ---

@st.cache_resource
def load_artifacts():
    """Loads the saved model and preprocessor."""
    preprocessor = joblib.load('preprocessor.joblib')
    model = joblib.load('model.joblib')
    return preprocessor, model

preprocessor, model = load_artifacts()

# --- 2. DEFINE THE APP'S UI (TITLE, SIDEBAR, INPUTS) ---

st.set_page_config(page_title="Profitability Predictor", layout="wide")

st.title("Transaction Profitability Predictor")
st.write("This app predicts whether a sales transaction is likely to be 'High Profit' or 'Standard Profit' based on its details. Input the transaction data in the sidebar to get a prediction.")

st.sidebar.header("Input Transaction Details")

countries = ['United States', 'United Kingdom', 'Germany', 'France']
product_categories = ['Accessories', 'Bikes', 'Clothing']
sub_categories = [
    'Tires and Tubes', 'Bottles and Cages', 'Helmets', 'Road Bikes', 
    'Mountain Bikes', 'Jerseys', 'Caps', 'Touring Bikes', 'Fenders', 
    'Shorts', 'Cleaners', 'Gloves', 'Hydration Packs', 'Socks', 'Vests', 
    'Bike Stands', 'Bike Racks'
]
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 
          'August', 'September', 'October', 'November', 'December']
genders = ['M', 'F']

# Input sidebar
unit_cost = st.sidebar.number_input('Unit Cost ($)', min_value=0.0, value=100.0, step=10.0)
unit_price = st.sidebar.number_input('Unit Price ($)', min_value=0.0, value=150.0, step=10.0)
quantity = st.sidebar.slider('Quantity', 1, 10, 2)
year = st.sidebar.selectbox('Year', [2015, 2016])
month = st.sidebar.selectbox('Month', months)
country = st.sidebar.selectbox('Country', countries)
product_category = st.sidebar.selectbox('Product Category', product_categories)
sub_category = st.sidebar.selectbox('Sub Category', sub_categories)
customer_age = st.sidebar.slider('Customer Age', 10, 90, 35)
customer_gender = st.sidebar.selectbox('Customer Gender', genders)


# --- 3. GATHER INPUTS AND PREPARE FOR PREDICTION ---

def create_input_df():
    age_bins = [0, 17, 25, 35, 50, 120]
    age_labels = ['Child', 'Youth', 'Young Adult', 'Adult', 'Senior']
    age_group = pd.cut([customer_age], bins=age_bins, labels=age_labels, right=False)[0]

    input_data = {
        'Year': [year],
        'Quantity': [quantity],
        'Unit Cost': [unit_cost],
        'Unit Price': [unit_price],
        'Month': [month],
        'Customer Gender': [customer_gender],
        'Country': [country],
        'Product Category': [product_category],
        'Sub Category': [sub_category],
        'Age_Group': [age_group],
        'Customer Age': [customer_age],
        'State': ['California'],
        'Date': [pd.to_datetime('today')]
    }
    
    expected_columns = [
        'Date', 'Year', 'Month', 'Customer Age', 'Customer Gender', 'Country', 
        'State', 'Product Category', 'Sub Category', 'Quantity', 'Unit Cost', 
        'Unit Price', 'Age_Group'
    ]
    
    input_df = pd.DataFrame(input_data)
    # Reorder columns to match the training order
    input_df = input_df.reindex(columns=expected_columns)
    
    return input_df

input_df = create_input_df()

# Display the user input for confirmation
st.subheader("User Input Summary")
st.dataframe(input_df)


# --- 4. PREDICT AND DISPLAY THE OUTPUT ---

if st.sidebar.button("Predict Profitability"):
    # 1. Preprocess the user input using the loaded pipeline
    input_prepared = preprocessor.transform(input_df)

    # 2. Make a prediction
    prediction = model.predict(input_prepared)
    prediction_proba = model.predict_proba(input_prepared)

    # 3. Display the result
    st.subheader("Prediction Result")
    
    if prediction[0] == 1:
        st.success("This transaction is predicted to be **High Profit**.")
        st.balloons()
    else:
        st.info("This transaction is predicted to be **Standard Profit**.")

    # 4. Display the prediction probabilities
    st.subheader("Prediction Probabilities")
    proba_df = pd.DataFrame({
        'Profit Category': ['Standard Profit (0)', 'High Profit (1)'],
        'Probability': prediction_proba[0]
    })
    st.bar_chart(proba_df.set_index('Profit Category'))