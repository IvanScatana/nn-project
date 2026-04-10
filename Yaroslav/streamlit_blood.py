import streamlit as st
from io import

st.title('Загрузи картинку чтобы определить тип клетки')

upload_file = st.file_uploader(
    'Выбери картинку в формате JPG, PNG, JPEG',
    accept_multiple_files=True,
    type=['']
)

if uploaded_files: