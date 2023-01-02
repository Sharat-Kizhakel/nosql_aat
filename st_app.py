import streamlit as st
import requests
from config import IP, PORT

st.title('COVID Data Explorer')
url = f'http://{IP}:{PORT}/'

def get_data():
    r = requests.get(url)
    data = r.json()

st.write(get_data())
