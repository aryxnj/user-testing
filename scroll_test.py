import streamlit as st
from datetime import datetime

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Function to add a new message
def add_message():
    st.session_state.messages.insert(0, f"New message at {datetime.now().strftime('%H:%M:%S')}")

st.title("Scroll-to-Top Simulation Without JavaScript")

# Button to add a new message
st.button("Add Message", on_click=add_message)

# Display messages
for msg in st.session_state.messages:
    st.write(msg)
