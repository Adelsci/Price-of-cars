import streamlit as st
import pandas as pd
import joblib

# Load your trained model
model = joblib.load('car_price_model.pkl')

# Load encoders and scaler
encoders = joblib.load('encoders.pkl')
scaler = joblib.load('scaler.pkl')

# Function to encode inputs
def encode_input(data, encoders):
    for column, encoder in encoders.items():
        data[column] = encoder.transform([data[column]])[0]
    return data
# Load the original dataset for visualizations
Car_listings = pd.read_csv('Car_listings.csv')
# Calculate IQR for Mileage and Price
Q1_mileage = Car_listings['Mileage'].quantile(0.25)
Q3_mileage = Car_listings['Mileage'].quantile(0.75)
IQR_mileage = Q3_mileage - Q1_mileage
lower_bound_mileage = Q1_mileage - 1.5 * IQR_mileage
upper_bound_mileage = Q3_mileage + 1.5 * IQR_mileage

Q1_price = Car_listings['Price'].quantile(0.25)
Q3_price = Car_listings['Price'].quantile(0.75)
IQR_price = Q3_price - Q1_price
lower_bound_price = Q1_price - 1.5 * IQR_price
upper_bound_price = Q3_price + 1.5 * IQR_price
# Filter out the outliers
data = Car_listings[(Car_listings['Mileage'] >= lower_bound_mileage) & (Car_listings['Mileage'] <= upper_bound_mileage) & 
                     (Car_listings['Price'] >= lower_bound_price) & (Car_listings['Price'] <= upper_bound_price)]

# Streamlit app
st.title('Car Price Prediction')

# Input fields for user
marke = st.selectbox('Marke', encoders['Marke'].classes_)
model_car = st.selectbox('Model', encoders['Model'].classes_)
year = st.slider('Year', 1997, 2018, 2014)
mileage = st.number_input('Mileage', min_value=0, max_value=300000, value=50000)
city = st.selectbox('City', encoders['City'].classes_)
state = st.selectbox('State', encoders['State'].classes_)

# Create a dataframe from the inputs
input_data = {'Marke': marke, 'Model': model_car, 'Year': year, 'Mileage': mileage, 'City': city, 'State': state}
input_df = pd.DataFrame([input_data])

# Encode and scale input data
input_df = encode_input(input_df, encoders)
input_df[['Mileage']] = scaler.transform(input_df[['Mileage']])

# Predict price
predicted_price = model.predict(input_df)[0]

st.write(f'Predicted Price: ${predicted_price:.2f}')
