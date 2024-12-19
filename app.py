import streamlit as st
from sqlalchemy import create_engine, text
from pathlib import Path
import random
from datetime import datetime

# Function to load local CSS
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file '{file_name}' not found. Please ensure it exists in the project directory.")

# Set page configuration FIRST
st.set_page_config(
    page_title="AI Music Assistant User Testing",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply custom CSS AFTER set_page_config
local_css("styles.css")  # Ensure you have a styles.css file in your project directory

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

if 'total_steps' not in st.session_state:
    # Calculate total steps: welcome + instructions + (inputs * outputs) + closing
    num_inputs = len(st.session_state.input_order)
    num_outputs = 4  # Assuming 4 models per input
    st.session_state.total_steps = 2 + (num_inputs * num_outputs) + 1  # welcome, instructions, closing

# Function to reset the session (for testing purposes)
def reset_session():
    for key in ['page', 'responses', 'current_input_index', 'current_output_index', 'input_order', 'output_orders', 'total_steps']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()  # Replace with st.rerun()

# Function to initialize database connection
def init_db():
    if 'db_engine' not in st.session_state:
        try:
            db_url = f"postgresql://{st.secrets['postgres']['user']}:{st.secrets['postgres']['password']}@" \
                     f"{st.secrets['postgres']['host']}:{st.secrets['postgres']['port']}/{st.secrets['postgres']['database']}"
            engine = create_engine(db_url, connect_args={"sslmode": st.secrets['postgres']['sslmode']})
            st.session_state.db_engine = engine
            # Removed the success message as per instructions
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
                elif response['page'] == 'feedback':
                    insert_query = text("""
                        INSERT INTO user_feedback (timestamp, feedback)
                        VALUES (:timestamp, :feedback)
                    """)
                    connection.execute(insert_query, {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'feedback': response.get('feedback', '')
                    })
        st.success("Your responses have been recorded. Thank you!")
    except Exception as e:
        st.error(f"Error saving responses: {e}")
        raise e  # Add this to display full error in Streamlit logs

# Welcome Page
def welcome_page():
    # Constrain the banner image width using a container
    with st.container():
        st.image("banner.png", use_container_width=True)  # Ensure 'banner.png' exists in your project directory
    st.title("🎵 Welcome to the AI Music Assistant User Testing 🎵")
    st.markdown("""
        Thank you for participating! In this study, you'll listen to original MIDI files and AI-generated continuations from various models. 
        Your feedback will help us enhance our AI's musical capabilities.
    """)

    with st.form("user_info"):
        st.subheader("Please provide some information about yourself:")
        col1, col2 = st.columns(2)
        with col1:
            musical_background = st.selectbox(
                "🎹 What is your musical background?",
                options=["Beginner", "Intermediate", "Advanced"]
            )
        with col2:
            age = st.text_input("🎂 Age:", "")
        gender = st.selectbox(
            "🧑 Gender:",
            options=["Prefer not to say", "Male", "Female", "Non-binary", "Other"]
        )
        submitted = st.form_submit_button("🚀 Start Testing")
        if submitted:
            # Validate age input
            if not age.isdigit() or not (5 < int(age) < 120):
                st.error("Please enter a valid age between 6 and 119.")
            else:
                # Save user info with timestamp
                st.session_state.responses.append({
                    'timestamp': datetime.now().isoformat(),
                    'page': 'welcome',
                    'musical_background': musical_background,
                    'age': age,
                    'gender': gender
                })
                st.session_state.page = 'instructions'
                st.rerun()  # Replace with st.rerun()

# Instructions Page
def instructions_page():
    st.title("📋 Testing Protocol Instructions 📋")
    st.markdown("""
        Here's how the testing protocol will work:
        
        1. **Listen to the Input MIDI:** You'll first hear an original MIDI file. This is the piece that the AI models will continue.
        
        2. **Listen to AI-Generated Continuations:** After the input, you'll hear several continuations generated by different AI models.
        
        3. **Rate Each Output:** For each AI-generated continuation, please rate its quality on a scale from 1 to 5, where 1 is the lowest and 5 is the highest.
        
        **Note:** The order of the MIDI files and the AI model outputs has been randomized to ensure unbiased feedback.
    """)
    if st.button("✅ Begin Testing"):
        st.session_state.page = 'testing'
        st.rerun()  # Replace with st.rerun()

# Testing Page
def testing_page():
    input_files = st.session_state.input_order
    current_input_index = st.session_state.current_input_index

    # Calculate current step for progress bar
    current_step = 2 + (current_input_index * 4) + st.session_state.current_output_index  # 2 for welcome and instructions
    progress = current_step / st.session_state.total_steps
    st.progress(progress)
    st.sidebar.markdown(f"**Progress:** Step {current_step} of {st.session_state.total_steps}")

    if current_input_index < len(input_files):
        current_input_file = input_files[current_input_index]
        input_name = current_input_file.name  # e.g., 'input-3.mp4'
        st.header(f"🔊 Listening to Input MIDI: {input_name.replace('.mp4', '').replace('-', ' ').capitalize()}")

        # Constrain input video within a container
        with st.container():
            st.video(str(current_input_file), start_time=0, format="video/mp4", help=None, key=None)

        outputs = st.session_state.output_orders[current_input_file.name]
        output_index = st.session_state.current_output_index
        total_outputs = len(outputs)

        if output_index < total_outputs:
            current_output = outputs[output_index]
            model_name = current_output['model']
            output_file = current_output['file']
            st.subheader(f"🎹 AI-Generated Continuation from {model_name.capitalize()} Model")

            # Constrain output video within a container
            with st.container():
                st.video(str(output_file), start_time=0, format="video/mp4", help=None, key=None)

            with st.form(f"rating_form_{current_input_index}_{output_index}"):
                # Replace slider with radio buttons for ratings
                rating = st.radio(
                    f"Rate the continuation from the **{model_name.capitalize()}** Model:",
                    options=[1, 2, 3, 4, 5],
                    index=2,
                    format_func=lambda x: "⭐" * x
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
                    st.success("✅ Rating submitted successfully!")
                    # Move to next output
                    st.session_state.current_output_index += 1
                    st.rerun()  # Replace with st.rerun()
        else:
            # Move to next input
            st.session_state.current_input_index += 1
            st.session_state.current_output_index = 0
            st.rerun()  # Replace with st.rerun()
    else:
        st.session_state.page = 'closing'
        st.rerun()  # Replace with st.rerun()

# Closing Page
def closing_page():
    # Constrain the closing banner image width using a container
    with st.container():
        st.image("closing_banner.png", use_container_width=True)  # Optional: Add a closing banner
    st.title("✅ Thank You for Your Participation! ✅")
    st.markdown("""
        We appreciate you taking the time to help us improve the AI Music Assistant. 
        Your feedback is invaluable and will contribute to the development of better musical tools.
    """)

    with st.form("additional_feedback"):
        feedback = st.text_area("📝 Do you have any additional comments or suggestions?", "")
        submitted = st.form_submit_button("📤 Submit and Exit")
        if submitted:
            if feedback:
                st.session_state.responses.append({
                    'timestamp': datetime.now().isoformat(),
                    'page': 'feedback',
                    'feedback': feedback
                })
            save_responses()
            st.stop()

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
