import streamlit as st
import pandas as pd
import pickle

with open(file="churn_final_data.pkl",mode="rb") as f:
    model = pickle.load(f)

st.title("Welcome to Telecom Churn Project")

st.sidebar.header('User Input Parameters')
   
def user_input_features():
    
    account_length = st.sidebar.number_input("Enter account length here")
    voice_mail_plan = st.sidebar.number_input("Enter voive mail plan here")
    voice_mail_messages = st.sidebar.number_input("Enter voice mail msg here")
    day_mins = st.sidebar.number_input("Enter day minutes here")
    evening_mins = st.sidebar.number_input("Enter evng min here")
    night_mins = st.sidebar.number_input("Enter night min here")
    international_mins = st.sidebar.number_input("Enter international min here")
    # customer_service_calls = st.sidebar.number_input("Enter service calls here")
    # international_plan = st.sidebar.number_input("Enter international plan here")
    day_calls = st.sidebar.number_input("Enter day calls here")
    day_charge = st.sidebar.number_input("Enter day charge here")
    evening_charge = st.sidebar.number_input("Enter evening calls here")
    night_charge = st.sidebar.number_input("Enter night charge here")
    # international_calls = st.sidebar.number_input("Enter international calls here")
    # international_charge = st.sidebar.number_input("Enter international charge here")
    total_charge = st.sidebar.number_input("Enter total charge here")
    data = {'account_length' : account_length,
            'voice_mail_plan' : voice_mail_plan,
            'voice_mail_messages' : voice_mail_messages,
            'day_mins ' : day_mins,
			'eve_mins' : evening_mins,
			'night_mins' : night_mins,
			'intl_mins' : international_mins,
			# 'customer_service_calls' : customer_service_calls,
			# 'international_plan' : international_plan,
			'day_calls' : day_calls,
			'day_charge' : day_charge,
			'eve_charge' : evening_charge,
			'night_charge' : night_charge,
			# 'international_calls' : international_calls,
			# 'international_charge' : international_charge,
			'total_charge' : total_charge}
    features = pd.DataFrame(data,index = [0])
    return features 

tel = user_input_features()
st.write(tel)



    
#st.write("Model loaded")

prediction = model.predict(tel)

prediction_proba = model.predict_proba(tel)

st.subheader('Prediction_Result')

st.write('*** CUSTOMER IS CHURN ***' if prediction_proba [0][1] > 0.5
         else '*** CUSTOMER WILL NOT CHURN ***')
