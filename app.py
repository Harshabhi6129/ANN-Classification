import streamlit as st 
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle
import pandas as pd
import numpy as np

#load the trained model
model = tf.keras.models.load_model('ann_model.keras')

#load the encoders and scalers
with open('OneHotEncoder_geo.pkl','rb') as file:
    OneHotEncoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

#Streamlit app
st.title = 'Customer Churn Prediction'

#User input
geography = st.selectbox('Geography', OneHotEncoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('estimated_salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_number = st.selectbox('Is Active Number', [0,1])

#prepare the input data
input_data = {
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure' : [tenure],
    'Balance':[balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_number],
    'EstimatedSalary': [estimated_salary]    
}

geo_encoded = OneHotEncoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=OneHotEncoder_geo.get_feature_names_out(['Geography']))

#combine one-hot encoded columns with encoded data
input_data = pd.DataFrame(input_data)
input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

#Scale the input data
input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

st.write(f'Churn Probability: {prediction_prob:.2f}')

if prediction_prob > 0.5:
    st.write('Customer is likely to churn')

else:
    st.write('Customer not likely to churn')


