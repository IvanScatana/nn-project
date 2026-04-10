import streamlit as st

st.title('Good evning!')

first_name = st.text_input("Write your name")
last_name = st.text_input("Write your family name")
if first_name == '' and last_name =='':
    message = 'Hello!'
else:
    message = f'Hello {first_name} {last_name}!'

st.write(message)

if 'show_text' not in st.session_state:
    st.session_state.show_text = False

def show_hander():
     st.session_state.show_text = not st.session_state.show_text

    
show_text = st.button('8> Click on me <8', on_click=show_hander())

if  st.session_state.show_text:
    st.write(f'I love you {first_name} {last_name}!')