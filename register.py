import streamlit as st
from db import insert_user

def run():
    st.title("Register Page")
    st.write("Please fill out the form to register.")

    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type='password')
    
    if st.button("Register"):
        if username and email and password:
            # Try to insert the user into the database
            if insert_user(username, email, password):
                st.success("Registration successful!")
            else:
                st.error("Username already exists. Please choose a different one.")
        else:
            st.error("All fields are required!")
