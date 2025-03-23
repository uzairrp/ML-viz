import streamlit as st
import pandas as pd
import pickle


st.set_page_config(page_title="Price Prediction", page_icon="💵")

# Loading the model and encoders
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('encoders.pkl', 'rb') as encoders_file:
    encoders = pickle.load(encoders_file)

features = ['manufacturer', 'body', 'drive', 'fuel_system', 'horsepower', 'city_mpg', 'highway_mpg', 'num_doors', 'num_cylinders']


st.title("Car Price Prediction")
st.subheader("In this page, the users can select the features of their choice to get a prediction.")
st.write("Enter the details of the car to predict its price")

# Input fields for each feature
# Encoded
manufacturer = st.selectbox('Manufacturer', encoders['manufacturer'].classes_)
body = st.selectbox('Body', encoders['body'].classes_)
drive = st.selectbox('Drive', encoders['drive'].classes_)
# Numerical
horsepower = st.number_input('Horsepower (Observerd minimum: 48)', min_value=48, step=1)
city_mpg = st.number_input('City MPG (Avg. 25)', min_value=1, step=1)
highway_mpg = st.number_input('Highway MPG (Avg. 30)', min_value=1, step=1)

num_doors = st.number_input('Number of Doors', min_value=2, max_value=5, step=1)  # Allowing for 2 to 5 doors
num_cylinders = st.number_input('Number of Cylinders (Observed minimum: 2)', min_value=2, max_value=12, step=2)  # 2 to 12 cylinders

# Use the most common value for `fuel_system`
fuel_system = 'spfi'
st.write(f"Fuel System: {fuel_system} (fixed)")

# Putting input data into a df for prediction
input_data = {
    'manufacturer': manufacturer,
    'body': body,
    'drive': drive,
    'fuel_system': fuel_system,
    'horsepower': horsepower,
    'city_mpg': city_mpg,
    'highway_mpg': highway_mpg,
    'num_doors': num_doors,
    'num_cylinders': num_cylinders
}
input_df = pd.DataFrame([input_data])


# Encoding 
for col in ['manufacturer', 'body', 'drive', 'fuel_system']:
    input_df[col] = encoders[col].transform(input_df[col])

# Making a prediction
if st.button('Predict'):
    prediction = model.predict(input_df)
    st.write(f"Predicted car price: ${prediction[0]:,.2f}")