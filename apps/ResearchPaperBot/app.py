from embedchain import App
from api import getKey
import os
import streamlit as st
import base64

api_key = getKey()
os.environ['OPENAI_API_KEY'] = api_key
app = App()

if not os.path.exists("Data"):
    os.makedirs("Data")

st.title("Research Paper Assistant")
pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if pdf_file is not None:
    pdf_contents = pdf_file.read()
    pdf_b64 = base64.b64encode(pdf_contents).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{pdf_b64}" width="500" height="800"></iframe>'
    st.sidebar.markdown(pdf_display, unsafe_allow_html=True)
    
    file_path = os.path.join("Data", pdf_file.name)
    with open(file_path, "wb") as f:
        f.write(pdf_file.getbuffer())
        
    app.add(file_path,data_type="pdf_file")
    prompt = st.text_input("Enter your query:")
    if prompt:
        with st.spinner("Generating..."):
            response = app.query(prompt)
            st.write(response)