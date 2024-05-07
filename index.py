import streamlit as st
import pandas as pd
import pickle
with open(file="churn_final_data.pkl",mode="rb") as f:
    model = pickle.load(f)

st.title("Welcome to Telecom Churn Project")
st.sidebar.header('User Input Parameters')
def user_input_features():    
    account_length = st.sidebar.number_input("Enter account length here")
    voice_plan = st.sidebar.number_input("Enter voive mail plan here")
    voice_messages = st.sidebar.number_input("Enter voice mail msg here")
    intl_plan = st.sidebar.number_input("Enter international plan here")
    intl_mins = st.sidebar.number_input("Enter international min here")
    intl_calls = st.sidebar.number_input("Enter international calls here")
    intl_charge = st.sidebar.number_input("Enter international charge here")
    day_mins = st.sidebar.number_input("Enter day minutes here")
    day_calls = st.sidebar.number_input("Enter day calls here")
    day_charge = st.sidebar.number_input("Enter day charge here")
    eve_mins = st.sidebar.number_input("Enter evening mins here")
    eve_calls = st.sidebar.number_input("Enter evening calls here")
    eve_charge = st.sidebar.number_input("Enter evening charge here")
    night_mins = st.sidebar.number_input("Enter Night mins here")
    night_calls = st.sidebar.number_input("Enter Night calls here")
    night_charge = st.sidebar.number_input("Enter Night charge here")
    customer_calls = st.sidebar.number_input("Enter number of calls here")
    # total_charge = st.sidebar.number_input("Enter total charge here")
    data = {'account_length' : account_length,
            'voice_plan' : voice_plan,
            'voice_messages' : voice_messages,
            'intl_plan' : intl_plan,
            'intl_mins' : intl_mins,
            'intl_calls' : intl_calls,
            'intl_charge' : intl_charge,
            'day_mins' : day_mins,
            'day_calls' : day_calls,
            'day_charge' : day_charge,
            'eve_mins' : eve_mins,
            'eve_calls' : eve_calls,
            'eve_charge' : eve_charge,
            'night_mins' : night_mins,
            'night_calls': night_calls,
            'night_charge' : night_charge,
            'customer_calls': customer_calls
            # 'total_charge' : total_charge
   }
    features = pd.DataFrame(data,index = [0])
    return features 

tel = user_input_features()
st.write(tel)

predictions = (model.predict(tel) > 0.5).astype("int32")

prediction_proba = model.predict_proba(tel)

st.subheader('Prediction_Result')

st.write('*** CUSTOMER IS CHURN ***' if prediction_proba [0][1] > 0.5
         else '*** CUSTOMER WILL NOT CHURN ***')
