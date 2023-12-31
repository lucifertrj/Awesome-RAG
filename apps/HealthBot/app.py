import streamlit as st
from embedchain import App
import pandas as pd
from api import getKey
import os

api_key = getKey()
os.environ['OPENAI_API_KEY'] = api_key
app = App()

st.title("Health Assistant Bot")
choice = st.sidebar.selectbox("Pick your choice",("Keto","Vegan"))
st.header(choice)

if choice == 'Keto':
    file_path = "Data/keto.csv"
    data = pd.read_csv(file_path)
elif choice == 'Vegan':
    file_path = "Data/vegan.csv"
    data = pd.read_csv(file_path)
else:
    pass
    
st.sidebar.dataframe(data.head(8))
app.add(file_path,data_type="csv")

prompt = st.text_input("Enter your query")
if prompt: 
    with st.spinner("Be healthy..."):
        response = app.chat(prompt)
        st.write(response)