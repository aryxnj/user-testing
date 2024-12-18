import streamlit as st
from sqlalchemy import create_engine, text
from pathlib import Path
import random
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="AI Music Assistant User Testing",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state variables
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'

if 'responses' not in st.session_state:
    st.session_state.responses = []

if 'current_input_index' not in st.session_state:
    st.session_state.current_input_index = 0

if 'current_output_index' not in st.session_state:
    st.session_state.current_output_index = 0

if 'input_order' not in st.session_state:
    # Define input files directory
    input_dir = Path("output_videos/")
    # List all input .mp4 files
    input_files = sorted([f for f in input_dir.glob("input-*.mp4")])
    # Shuffle the input order
    random.shuffle(input_files)
    st.session_state.input_order = input_files

if 'output_orders' not in st.session_state:
    # Define the models based on the suffixes in output filenames
    models = ['attention', 'basic', 'lookback', 'mono']
    output_dir = Path("output_videos/")
    output_orders = {}
    for input_file in st.session_state.input_order:
        input_name = input_file.stem.split('-')[1]  # Extract the number, e.g., '3' from 'input-3'
        # Collect all outputs for this input across models
        outputs = []
        for model in models:
            output_file = output_dir / f"output-{input_name}-{model}.mp4"
            if output_file.exists():
                outputs.append({
                    'model': model,
                    'file': output_file
                })
            else:
                st.warning(f"Missing output file: {output_file.name}")
        # Shuffle the outputs for this input
        random.shuffle(outputs)
        output_orders[input_file.name] = outputs
    st.session_state.output_orders = output_orders

# Function to reset the session (for testing purposes)
def reset_session():
    for key in ['page', 'responses', 'current_input_index', 'current_output_index', 'input_order', 'output_orders']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# Function to initialize database connection
def init_db():
    if 'db_engine' not in st.session_state:
        try:
            db_url = f"postgresql://{st.secrets['postgres']['user']}:{st.secrets['postgres']['password']}@" \
                     f"{st.secrets['postgres']['host']}:{st.secrets['postgres']['port']}/{st.secrets['postgres']['database']}"
            engine = create_engine(db_url, connect_args={"sslmode": st.secrets['postgres']['sslmode']})
            st.session_state.db_engine = engine
            st.success("Database connection established successfully.")
        except Exception as e:
            st.error(f"Error connecting to PostgreSQL: {e}")

# Function to save responses to PostgreSQL
def save_responses():
    if 'db_engine' not in st.session_state:
        st.error("Database not initialized.")
        return
    engine = st.session_state.db_engine
    try:
        with engine.begin() as connection:  # Use engine.begin to auto-commit
            for response in st.session_state.responses:
                if response['page'] == 'welcome':
                    insert_query = text("""
                        INSERT INTO user_info (timestamp, musical_background, age, gender)
                        VALUES (:timestamp, :musical_background, :age, :gender)
                    """)
                    connection.execute(insert_query, {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'musical_background': response.get('musical_background', ''),
                        'age': response.get('age', ''),
                        'gender': response.get('gender', '')
                    })
                elif response['page'] == 'testing':
                    insert_query = text("""
                        INSERT INTO user_ratings (timestamp, input_file, output_file, model, rating)
                        VALUES (:timestamp, :input_file, :output_file, :model, :rating)
                    """)
                    connection.execute(insert_query, {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'input_file': response.get('input', ''),
                        'output_file': response.get('output', ''),
                        'model': response.get('model', ''),
                        'rating': response.get('rating', 0)
                    })
        st.success("Your responses have been recorded. Thank you!")
    except Exception as e:
        st.error(f"Error saving responses: {e}")
        raise e  # Add this to display full error in Streamlit logs

        
def test_db_connection():
    if 'db_engine' not in st.session_state:
        st.error("Database not initialized.")
        return
    engine = st.session_state.db_engine
    try:
        with engine.connect() as connection:
            # Insert a test row into user_info
            insert_query = text("""
                INSERT INTO user_info (timestamp, musical_background, age, gender)
                VALUES (:timestamp, :musical_background, :age, :gender)
            """)
            connection.execute(insert_query, {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'musical_background': 'Test Background',
                'age': '99',
                'gender': 'Test Gender'
            })
            st.success("Test row inserted successfully into user_info.")
    except Exception as e:
        st.error(f"Error during test insert: {e}")


# Welcome Page
def welcome_page():
    st.title("🎵 Welcome to the AI Music Assistant User Testing 🎵")
    st.write("""
        Thank you for participating in our user testing for the AI Music Assistant. 
        In this study, you'll listen to original MIDI files and their AI-generated continuations from various models. 
        Your feedback will help us improve our models and provide better musical experiences.
    """)
    
    with st.form("user_info"):
        st.subheader("Please provide some information about yourself:")
        musical_background = st.selectbox(
            "What is your musical background?",
            options=["Beginner", "Intermediate", "Advanced"]
        )
        age = st.text_input("Age:", "")
        gender = st.selectbox(
            "Gender:",
            options=["Prefer not to say", "Male", "Female", "Non-binary", "Other"]
        )
        submitted = st.form_submit_button("Start Testing")
        if submitted:
            # Save user info with timestamp
            st.session_state.responses.append({
                'timestamp': datetime.now().isoformat(),
                'page': 'welcome',
                'musical_background': musical_background,
                'age': age,
                'gender': gender
            })
            st.session_state.page = 'instructions'
            st.rerun()

# Instructions Page
def instructions_page():
    st.title("📋 Testing Protocol Instructions 📋")
    st.write("""
        Here's how the testing protocol will work:
        
        1. **Listen to the Input MIDI:** You'll first hear an original MIDI file. This is the piece that the AI models will continue.
        
        2. **Listen to AI-Generated Continuations:** After the input, you'll hear several continuations generated by different AI models.
        
        3. **Rate Each Output:** For each AI-generated continuation, please rate its quality on a scale from 1 to 5, where 1 is the lowest and 5 is the highest.
        
        **Note:** The order of the MIDI files and the AI model outputs has been randomized to ensure unbiased feedback.
    """)
    if st.button("Begin Testing"):
        st.session_state.page = 'testing'
        st.rerun()

# Testing Page
def testing_page():
    input_files = st.session_state.input_order
    current_input_index = st.session_state.current_input_index
    
    if current_input_index < len(input_files):
        current_input_file = input_files[current_input_index]
        input_name = current_input_file.name  # e.g., 'input-3.mp4'
        st.header(f"🔊 Listening to Input MIDI: {input_name.replace('.mp4', '').replace('-', ' ').capitalize()}")
        st.video(str(current_input_file))
        
        outputs = st.session_state.output_orders[current_input_file.name]
        output_index = st.session_state.current_output_index
        total_outputs = len(outputs)
        
        if output_index < total_outputs:
            current_output = outputs[output_index]
            model_name = current_output['model']
            output_file = current_output['file']
            st.subheader(f"🎹 AI-Generated Continuation from {model_name.capitalize()} Model")
            st.video(str(output_file))
            
            with st.form(f"rating_form_{current_input_index}_{output_index}"):
                rating = st.slider(
                    f"Rate the continuation from {model_name.capitalize()} Model (1 - Poor, 5 - Excellent):",
                    min_value=1, max_value=5, value=3, step=1
                )
                submitted = st.form_submit_button("Submit Rating")
                if submitted:
                    # Save the rating with timestamp
                    st.session_state.responses.append({
                        'timestamp': datetime.now().isoformat(),
                        'page': 'testing',
                        'input': current_input_file.name,
                        'output': output_file.name,
                        'model': model_name,
                        'rating': rating
                    })
                    # Move to next output
                    st.session_state.current_output_index += 1
                    st.rerun()
        else:
            # Move to next input
            st.session_state.current_input_index += 1
            st.session_state.current_output_index = 0
            st.rerun()
    else:
        st.session_state.page = 'closing'
        st.rerun()

# Closing Page
def closing_page():
    st.title("✅ Thank You for Your Participation! ✅")
    st.write("""
        We appreciate you taking the time to help us improve the AI Music Assistant. 
        Your feedback is invaluable and will contribute to the development of better musical tools.
    """)
    if st.button("Submit Responses and Exit"):
        save_responses()
        st.stop()
    if st.button("Restart"):
        reset_session()

# Initialize Database Connection
init_db()

# Navigation
if st.session_state.page == 'welcome':
    welcome_page()
elif st.session_state.page == 'instructions':
    instructions_page()
elif st.session_state.page == 'testing':
    testing_page()
elif st.session_state.page == 'closing':
    closing_page()
else:
    st.error("Unknown page!")

# Debug Section: Show collected responses (for troubleshooting)
if st.checkbox("Show Collected Responses"):
    st.write(st.session_state.responses)

# Optionally, run the database connection test
if st.checkbox("Run Database Connection Test"):
    test_db_connection()

