# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 15:40:45 2022

@author: Travail
"""

import streamlit as st
import requests
import pickle, dill
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler as mms
import bz2file as bz2

img = cv2.imread('homecred.png')

def decompress_pickle(file):
    data = bz2.BZ2File(file, "rb")
    data = pickle.load(data)
    return data

st.image(img)
st.title('Home Credit prediction App')

index = requests.get('https://homecred.herokuapp.com/login')
index_page = index.text
st.header(index_page)

client_list = pickle.load(open('customer_list.pickle', 'rb'))
lime = dill.load(open('lime.dill','rb'))
df = decompress_pickle("test_df.pbz2")
imputer = pickle.load(open('impute.pickle', 'rb'))
scaler = pickle.load(open('scale.pickle', 'rb'))
model = pickle.load(open('model.pickle', 'rb'))
clients_all = pickle.load(open('clients.pickle','rb'))


df = df.reset_index(drop = True)
df_cols = df.columns
clients_all = pd.DataFrame(clients_all)
clients_all = clients_all.reset_index(drop=True)

guide = pickle.load(open('guide.pickle','rb'))


cust_id = st.selectbox('Client IDs list', client_list)

#welcoming
url = 'http://localhost:5000/login/' + str(cust_id)
cust_hub = requests.get(url)
welcome = cust_hub.text
st.subheader('Your predictions for client ID : '+ welcome)

#prediction booth
url_prediction = 'https://homecred.herokuapp.com/login/predict/' + str(cust_id)
prediction = requests.get(url_prediction)
result = prediction.text

if float(result) >= 0.60:
    st.success('CREDIT GRANTED !')
else:
    st.error('CREDIT DENIED !')
    
st.header('Credit refund probability : ' + result)
st.write('Delta with probability threshold p=(0.60) : ' + str(round(0.60-float(result),2)))

#feature importance
st.subheader('Feature importance for this prediction')
feat_importances = pd.Series(model.feature_importances_, index=df_cols)
st.bar_chart(feat_importances.nlargest(20))

#lime-df imputing+scaling to get values
st.subheader('Local explanation')
df_2 = df.copy()
df_2 = imputer.transform(df_2)
df_2 = scaler.transform(df_2)
df_2 = pd.DataFrame(df_2)
indexing = clients_all[clients_all.SK_ID_CURR == cust_id].index[0]
exp = lime.explain_instance(df_2.iloc[indexing,:],model.predict_proba,num_features=20)
exp = exp.as_pyplot_figure()
st.pyplot(exp)

#general interpretability
sub_df = pd.DataFrame(df_2)
sub_df.columns = df_cols
st.subheader('General explanation')
feat_list = feat_importances.nlargest(20).index.tolist()
gen_choice = st.selectbox('Select feature', feat_list)
gen_choice = "%s" % (gen_choice)

if gen_choice == 'DAYS_EMPLOYED_PERC':
    desc = 'Ratio of days of employment on the whole life of the client'
    spec = 'None'
elif gen_choice == 'INCOME_CREDIT_PERC':
    desc = 'Ratio of total income on the ammount of the asked credit'
    spec = 'None'
elif gen_choice == 'INCOME_PER_PERSON':
    desc = 'Ratio of total house income per person living on it'
    spec = 'None'
elif gen_choice == 'ANNUITY_INCOME_PERC':
    desc = 'Ratio of annuity paid on the total house income'
    spec = 'None'
elif gen_choice == 'PAYMENT_RATE':
    desc = 'Ratio of the annual sum paid on the total ammount of the asked credit'
    spec = 'None'
else:
    desc = guide.loc[guide.Row == gen_choice, 'Description'].iloc[0]
    spec = guide.loc[guide.Row == gen_choice, 'Special'].iloc[0]
st.write ("Feature name : %s" % (gen_choice))
st.write("Description : %s" % (desc))
st.write("Additional notes : %s" % (spec))

df_non_scaled = df.copy()
df_non_scaled = imputer.transform(df_non_scaled)
df_non_scaled = pd.DataFrame(df_non_scaled)
df_non_scaled.columns = df_cols


#plot
gen_exp = plt.figure(figsize = (8,5))
sns.distplot(df_non_scaled[gen_choice])
plt.axvline(x = df_non_scaled.loc[indexing,gen_choice], color='red')
st.pyplot(gen_exp)



        
        
        
        
        
        