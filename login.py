import streamlit as st
from db import validate_user

def run():
    # Initialize session state for tracking login status if not already present
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = ""

    st.title("Login Page")
    st.write("Please log in to access the platform.")

    # Display username and password inputs
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')

    # Display login button
    if st.button("Login"):
        if not username or not password:
            # If either field is empty, show an error message
            st.error("Both username and password are required!")
        else:
            # Validate user credentials from the database
            user = validate_user(username, password)
            if user:
                # Successful login
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome, {username}!")
                st.session_state.page = "Prediction"  # Set page to Prediction
                st.rerun()  # Refresh the page to load Prediction page
            else:
                # If credentials are incorrect
                st.error("Invalid username or password. Please try again.")
    
    # Optionally, add a logout button
    if st.session_state.logged_in:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.page = "Login"  # Set page to Login after logout
            st.rerun()  # Refresh the page after logout
