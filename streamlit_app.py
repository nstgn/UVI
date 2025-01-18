import streamlit as st
from streamlit_gsheets import GSheetsConnection

url = "https://docs.google.com/spreadsheets/d/1SczaIV1JHUSca1hPilByJFFzOi5a8Hkhi0OemlmPQsY/edit?usp=sharing"
conn = st.connection("gsheets", type=GSheetsConnection)

#data = conn.read(worksheet="Sheet1")
data = conn.read(spreadsheet=url, usecols=[0, 1, 2, 3])
st.dataframe(data)
