import streamlit as st
from sqlalchemy import create_engine, text
from pathlib import Path
import random
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="AI Music Assistant User Testing",
    layout="centered",
    initial_sidebar_state="expanded"
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
    # List all input .mp4 files, excluding the Atonal, Wide-Range Melody (input-5.mp4)
    input_files = sorted([f for f in input_dir.glob("input-*.mp4") if f.name != "input-5.mp4"])
    # Shuffle the input order
    random.shuffle(input_files)
    st.session_state.input_order = input_files

if 'output_orders' not in st.session_state:
    # Define the models based on the suffixes in output filenames
    models = ['attention', 'basic', 'lookback', 'mono', 'lstm', 'markov']
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
    # Calculate total steps: (inputs * outputs) = total ratings
    num_inputs = len(st.session_state.input_order)
    num_outputs = 6
    st.session_state.total_steps = num_inputs * num_outputs  # e.g., 18 ratings

# Function to reset the session
def reset_session():
    for key in ['page', 'responses', 'current_input_index', 'current_output_index', 'input_order', 'output_orders', 'total_steps']:
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
        except Exception as e:
            st.error(f"Error connecting to PostgreSQL: {e}")

# Function to save user information and ratings to PostgreSQL with spinner
def save_ratings():
    if 'db_engine' not in st.session_state:
        st.error("Database not initialized.")
        return
    engine = st.session_state.db_engine
    try:
        with st.spinner("Saving your ratings..."):
            with engine.begin() as connection:  # Use engine.begin to auto-commit
                # Save user_info
                for response in st.session_state.responses:
                    if response['page'] == 'welcome':
                        insert_query = text("""
                            INSERT INTO user_info (timestamp, musical_background, age, gender)
                            VALUES (:timestamp, :musical_background, :age, :gender)
                        """)
                        connection.execute(insert_query, {
                            'timestamp': response.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                            'musical_background': response.get('musical_background', ''),
                            'age': response.get('age', ''),
                            'gender': response.get('gender', '')
                        })
                # Save user_ratings
                for response in st.session_state.responses:
                    if response['page'] == 'testing':
                        insert_query = text("""
                            INSERT INTO user_ratings (timestamp, input_file, output_file, continuation_number, model, criterion, rating)
                            VALUES (:timestamp, :input_file, :output_file, :continuation_number, :model, :criterion, :rating)
                        """)
                        connection.execute(insert_query, {
                            'timestamp': response.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                            'input_file': response.get('input', ''),
                            'output_file': response.get('output', ''),
                            'continuation_number': response.get('continuation_number', 0),
                            'model': response.get('model', ''),
                            'criterion': response.get('criterion', ''),
                            'rating': response.get('rating', 0)
                        })
        st.success("✅ Your information and ratings have been successfully saved. Thank you!")
    except Exception as e:
        st.error(f"Error saving ratings: {e}")

# Function to save feedback to PostgreSQL with spinner
def save_feedback():
    if 'db_engine' not in st.session_state:
        st.error("Database not initialized.")
        return
    engine = st.session_state.db_engine
    try:
        with st.spinner("Submitting your feedback..."):
            with engine.begin() as connection:
                for response in st.session_state.responses:
                    if response['page'] == 'feedback':
                        insert_query = text("""
                            INSERT INTO user_feedback (timestamp, feedback)
                            VALUES (:timestamp, :feedback)
                        """)
                        connection.execute(insert_query, {
                            'timestamp': response.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                            'feedback': response.get('feedback', '')
                        })
        st.success("✅ Your feedback has been submitted successfully.")
    except Exception as e:
        st.error(f"Error saving feedback: {e}")

# Mapping of input files to their descriptive names
input_name_mapping = {
    "input-3.mp4": "Familiar Tonal Snippet",
    "input-4.mp4": "Medium-Length Original Melody",
    "input-6.mp4": "Long Repetitive Motif"
}

# Evaluation Criteria
evaluation_criteria = [
    {
        "name": "Pitch/Tonal Coherence",
        "description": "How well does the continuation fit into an implied key or tonal centre established by the primer melody (if any)? For atonal examples, does the model at least maintain consistent intervallic relationships or avoid overly jarring leaps without intent?",
        "scoring": {
            "1": "Completely random, no sense of pitch organisation",
            "3": "Generally consistent pitches, occasional off-key notes if tonal context applies",
            "5": "Strong tonal coherence, notes feel harmonically related or purposefully atonal but structured"
        }
    },
    {
        "name": "Rhythmic Stability & Flow",
        "description": "Does the continuation maintain a clear rhythmic pulse and align well with the time signature and feel of the initial segment? Is it rhythmically logical, or too erratic and disjointed?",
        "scoring": {
            "1": "Erratic durations, awkward pauses, no clear rhythmic pattern",
            "3": "Some stability, but occasional unnatural note placements",
            "5": "Flows naturally, rhythmically coherent, and matches or complements the original rhythm"
        }
    },
    {
        "name": "Motivic & Thematic Continuity",
        "description": "Does the continuation pick up on melodic motifs, patterns, or contours from the original melody and develop them further? Or does it abruptly shift to unrelated ideas?",
        "scoring": {
            "1": "Completely unrelated sequence, abrupt changes with no continuity",
            "3": "Some fragments of previous patterns appear, but not consistently",
            "5": "Smooth thematic development, clear recognition and transformation of initial motifs"
        }
    },
    {
        "name": "Range & Complexity Management",
        "description": "How well does the model manage pitch range and complexity? Does it stay within a logical register, or jump erratically across octaves? Is the complexity (interval sizes, rhythmic subdivisions) appropriate for the style?",
        "scoring": {
            "1": "Extreme register jumps without reason, overly complex or simplistic to the point of incoherence",
            "3": "Generally appropriate complexity, a few overly large leaps or simplistic streaks",
            "5": "Balanced complexity, maintains a comfortable range and adds variety without chaos"
        }
    },
    {
        "name": "Aesthetic/Subjective Appeal",
        "description": "A more subjective measure of how pleasant, engaging, or musically “satisfying” the continuation sounds.",
        "scoring": {
            "1": "Unpleasant, jarring, or meaningless sequence",
            "3": "Acceptable but somewhat bland or directionless",
            "5": "Musically appealing, has a sense of purpose or emotional quality"
        }
    }
]

# Function to render the sidebar
def render_sidebar():
    st.sidebar.title("📋 Contents")
    st.sidebar.markdown("---")
    
    pages = ["welcome", "instructions", "testing", "closing"]
    for page in pages:
        display_name = page.capitalize()
        if st.session_state.page == page:
            # Highlight current page
            st.sidebar.markdown(f"### **{display_name}**")
        else:
            st.sidebar.markdown(f"{display_name}")
    
    st.sidebar.markdown("---")
    
    # Only show Testing Progress or Completed when in Testing-related pages
    if st.session_state.page in ['testing', 'closing']:
        st.sidebar.subheader("📝 Testing Progress")
        if st.session_state.page == 'closing':
            # Completed
            st.sidebar.markdown("Completed ✅")
            st.sidebar.progress(1.0)
        else:
            total_continuations = st.session_state.total_steps
            # Completed continuations
            completed_continuations = len({
                (resp['input'], resp['output'])
                for resp in st.session_state.responses
                if resp['page'] == 'testing'
            })
            progress_text = f"{completed_continuations}/{total_continuations} Continuations Completed"
            st.sidebar.text(progress_text)
            progress_percentage = completed_continuations / total_continuations if total_continuations else 0
            st.sidebar.progress(progress_percentage)
        
        st.sidebar.markdown("---")
    
    # Reset Session Button
    if st.sidebar.button("🔄 Reset Session"):
        reset_session()

# Welcome Page
def welcome_page():
    st.image("banner.png", use_container_width=True)
    st.title("🎵 Welcome to the AI Music Assistant User Testing 🎵")
    st.markdown("""
        Thank you for participating! In this study, you'll listen to original MIDI files and continuations from various models. 
        Your feedback will help us enhance our AI's musical capabilities.
    """)
    
    with st.form("user_info"):
        st.subheader("Please provide some information about yourself:")
        col1, col2 = st.columns(2)
        with col1:
            musical_background = st.selectbox(
                "🎹 What is your musical background?",
                options=["Select", "Beginner", "Intermediate", "Advanced"],
                index=0
            )
        with col2:
            age = st.text_input("🎂 Age:", "")
        gender = st.selectbox(
            "🧑 Gender:",
            options=["Select", "Male", "Female", "Non-binary", "Other"],
            index=0
        )
        submitted = st.form_submit_button("🚀 Start Testing")
        if submitted:
            # Validate inputs
            if not age.isdigit():
                st.error("Please enter a valid age.")
            elif musical_background == "Select":
                st.error("Please select your musical background.")
            elif gender == "Select":
                st.error("Please select your gender.")
            else:
                # Save user info
                st.session_state.responses.append({
                    'timestamp': datetime.now().isoformat(),
                    'page': 'welcome',
                    'musical_background': musical_background,
                    'age': age,
                    'gender': gender
                })
                # Move to instructions page
                st.session_state.page = 'instructions'
                st.rerun()

# Instructions Page
def instructions_page():
    st.title("📋 Testing Protocol Instructions 📋")
    st.markdown("""
        ### Scoring Method

        - For each model’s generated continuation of each test melody, assign a score of **1 to 5** for each criterion.
        - **Total the scores** for a composite result, or consider different weights for each criterion depending on the experiment’s priorities.
        - **Compare results** across models, parameter settings, and input melodies to draw conclusions about which configurations yield the most coherent, appealing, and stylistically appropriate continuations.

        ### Test Melodies Overview

        Each melody has been designed to represent different musical characteristics, complexity levels, and lengths. The idea is to provide a diverse set of test inputs so the continuation models can be evaluated on a range of scenarios.

        1. **Familiar Tonal Snippet**
            - **Description:** A short excerpt inspired by a familiar children’s melody, using only a few pitches within a simple scale.
            - **Reasoning:** Evaluates how well the model continues a well-known melodic pattern, potentially revealing if it respects common tonal tendencies and phrase endings.

        2. **Medium-Length Original Melody**
            - **Description:** A custom, moderately long melody that mixes different note lengths and steps mostly within a major scale.
            - **Reasoning:** Provides a more realistic musical context, testing the model’s handling of basic musical structure, varied rhythms, and thematic development.

        3. **Long Repetitive Motif**
            - **Description:** An extended sequence made by repeating a two-bar motif several times.
            - **Reasoning:** Assesses how the model deals with longer context and repetitive patterns. Will it continue with variations, remain consistent, or diverge unexpectedly?
    """)
    if st.button("✅ Begin Testing"):
        st.session_state.page = 'testing'
        st.rerun()

# Testing Page
def testing_page():
    input_files = st.session_state.input_order
    current_input_index = st.session_state.current_input_index

    if current_input_index < len(input_files):
        current_input_file = input_files[current_input_index]
        input_name = current_input_file.name  # e.g., 'input-3.mp4'
        descriptive_name = input_name_mapping.get(input_name, input_name)
        
        # Fetch outputs for the current input
        outputs = st.session_state.output_orders[current_input_file.name]
        output_index = st.session_state.current_output_index
        total_outputs = len(outputs)

        if output_index < total_outputs:
            current_output = outputs[output_index]
            continuation_number = output_index + 1  # 1 to 4
            output_file = current_output['file']
            model_name = current_output['model'].capitalize()  # e.g., 'Attention'

            st.header(f"🔊 Listening to Input MIDI: {descriptive_name}")

            # Conditional Rendering based on Continuation Number
            if continuation_number == 1:
                # For Continuation 1, display the input video normally
                st.video(str(current_input_file))
            else:
                # For Continuations 2-4, display the input video inside a closed expander
                with st.expander(f"🔍 View Input MIDI for Continuation {continuation_number}", expanded=False):
                    st.video(str(current_input_file))

            st.subheader(f"🎹 Continuation {continuation_number} - Model: {model_name}")
            st.video(str(output_file))

            with st.form(f"rating_form_{current_input_index}_{output_index}"):
                st.markdown("**Please rate the following criteria:**")
                
                # Create tabs for each evaluation criterion
                tab_names = [criterion['name'] for criterion in evaluation_criteria]
                tabs = st.tabs(tab_names)

                ratings = {}  # Dictionary to hold ratings for each criterion
                unanswered_criteria = []  # List to track unanswered criteria

                for tab, criterion in zip(tabs, evaluation_criteria):
                    with tab:
                        st.markdown(f"*{criterion['description']}*")
                        st.markdown("**Scoring:**")
                        st.markdown(f"- **1:** {criterion['scoring']['1']}")
                        st.markdown(f"- **3:** {criterion['scoring']['3']}")
                        st.markdown(f"- **5:** {criterion['scoring']['5']}")
                        
                        # Select box for rating with 'Select' as default
                        rating = st.selectbox(
                            f"Rate {criterion['name']}:",
                            options=['Select', 1, 2, 3, 4, 5],
                            index=0,
                            key=f"{current_input_index}_{output_index}_{criterion['name']}"
                        )
                        
                        # Store the rating in the dictionary
                        ratings[criterion['name']] = rating

                submitted = st.form_submit_button("Submit Ratings")
                if submitted:
                    # Check for unanswered criteria
                    for criterion_name, rating in ratings.items():
                        if rating == 'Select':
                            unanswered_criteria.append(criterion_name)
                    
                    if unanswered_criteria:
                        st.error("Please rate all criteria before submitting.")
                        st.warning(f"Unanswered Criteria: {', '.join(unanswered_criteria)}")
                    else:
                        # Append all ratings at once to avoid multiple entries
                        for criterion_name, rating in ratings.items():
                            st.session_state.responses.append({
                                'timestamp': datetime.now().isoformat(),
                                'page': 'testing',
                                'input': current_input_file.name,
                                'output': output_file.name,
                                'continuation_number': continuation_number,
                                'model': model_name,
                                'criterion': criterion_name,
                                'rating': rating
                            })
                        st.success("✅ Ratings submitted successfully!")
                        # Move to next output
                        st.session_state.current_output_index += 1
                        st.rerun()
        
            # Debug Submit Button
            st.markdown("---")
            if st.button("Debug Submit"):
                for criterion in evaluation_criteria:
                    st.session_state.responses.append({
                        'timestamp': datetime.now().isoformat(),
                        'page': 'testing',
                        'input': current_input_file.name,
                        'output': output_file.name,
                        'continuation_number': continuation_number,
                        'model': model_name,
                        'criterion': criterion['name'],
                        'rating': 3
                    })
                st.success("✅ Ratings submitted with default values (3).")
                # Move to next output
                st.session_state.current_output_index += 1
                st.rerun()
        else:
            # All outputs for the current input have been rated
            # Check if it's the last input
            if st.session_state.current_input_index + 1 >= len(input_files):
                # All inputs have been processed, save ratings and move to closing
                save_ratings()
                st.session_state.page = 'closing'
                st.rerun()
            else:
                # Move to next input
                st.session_state.current_input_index += 1
                st.session_state.current_output_index = 0
                st.rerun()
    else:
        st.session_state.page = 'closing'
        st.rerun()

# Closing Page with Balloons
def closing_page():
    st.image("closing_banner.png", use_container_width=True)
    st.title("✅ Thank You for Your Participation!")
    st.markdown("""
        We appreciate you taking the time to help us improve the AI Music Assistant. 
        Your feedback is invaluable and will contribute to the development of better musical tools.
    """)

    # Bfalloons effect after saving feedback
    st.balloons()
    
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
            # Save feedback to the database
            save_feedback()
            st.stop()

# Initialize Database Connection
init_db()

# Render the sidebar
render_sidebar()

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
