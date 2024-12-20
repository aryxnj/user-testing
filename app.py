import streamlit as st
from sqlalchemy import create_engine, text
from pathlib import Path
import random
from datetime import datetime

# Set page configuration with a centered layout
st.set_page_config(
    page_title="AI Music Assistant User Testing",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Initialize session state variables
def initialize_session():
    if 'responses' not in st.session_state:
        st.session_state.responses = []
    
    if 'current_input_index' not in st.session_state:
        st.session_state.current_input_index = 0
    
    if 'current_output_index' not in st.session_state:
        st.session_state.current_output_index = 0
    
    if 'input_order' not in st.session_state:
        input_dir = Path("output_videos/")
        input_files = sorted([f for f in input_dir.glob("input-*.mp4")])
        random.shuffle(input_files)
        st.session_state.input_order = input_files
    
    if 'output_orders' not in st.session_state:
        models = ['attention', 'basic', 'lookback', 'mono']
        output_dir = Path("output_videos/")
        output_orders = {}
        for input_file in st.session_state.input_order:
            input_name = input_file.stem.split('-')[1]
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
            random.shuffle(outputs)
            output_orders[input_file.name] = outputs
        st.session_state.output_orders = output_orders
    
    if 'total_steps' not in st.session_state:
        num_inputs = len(st.session_state.input_order)
        num_outputs = 4  # Assuming 4 models per input
        st.session_state.total_steps = 2 + (num_inputs * num_outputs) + 1  # welcome, instructions, closing

# Call the session initializer
initialize_session()

# Function to reset the session (for testing purposes)
def reset_session():
    for key in ['responses', 'current_input_index', 'current_output_index', 'input_order', 'output_orders', 'total_steps']:
        if key in st.session_state:
            del st.session_state[key]
    st.experimental_rerun()

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

# Call the database initializer
init_db()

# Function to save user information and ratings to PostgreSQL
def save_ratings():
    if 'db_engine' not in st.session_state:
        st.error("Database not initialized.")
        return
    engine = st.session_state.db_engine
    try:
        with engine.begin() as connection:
            # Save user_info
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
            # Save user_ratings
            for response in st.session_state.responses:
                if response['page'] == 'testing':
                    insert_query = text("""
                        INSERT INTO user_ratings (timestamp, input_file, output_file, continuation_number, criterion, rating)
                        VALUES (:timestamp, :input_file, :output_file, :continuation_number, :criterion, :rating)
                    """)
                    connection.execute(insert_query, {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'input_file': response.get('input', ''),
                        'output_file': response.get('output', ''),
                        'continuation_number': response.get('continuation_number', 0),
                        'criterion': response.get('criterion', ''),
                        'rating': response.get('rating', 0)
                    })
    except Exception as e:
        st.error(f"Error saving ratings: {e}")
        raise e

# Function to save feedback to PostgreSQL
def save_feedback():
    if 'db_engine' not in st.session_state:
        st.error("Database not initialized.")
        return
    engine = st.session_state.db_engine
    try:
        with engine.begin() as connection:
            for response in st.session_state.responses:
                if response['page'] == 'feedback':
                    insert_query = text("""
                        INSERT INTO user_feedback (timestamp, feedback)
                        VALUES (:timestamp, :feedback)
                    """)
                    connection.execute(insert_query, {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'feedback': response.get('feedback', '')
                    })
    except Exception as e:
        st.error(f"Error saving feedback: {e}")
        raise e

# Mapping of input files to their descriptive names
input_name_mapping = {
    "input-3.mp4": "Familiar Tonal Snippet",
    "input-4.mp4": "Medium-Length Original Melody",
    "input-5.mp4": "Atonal, Wide-Range Melody",
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
    # ... (other criteria remain unchanged)
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

# Define tabs for navigation
tabs = st.tabs(["📋 Welcome", "📝 Instructions", "🎵 Testing", "✅ Closing"])

# ---- Welcome Tab ----
with tabs[0]:
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
                # Save user info with timestamp
                st.session_state.responses.append({
                    'timestamp': datetime.now().isoformat(),
                    'page': 'welcome',
                    'musical_background': musical_background,
                    'age': age,
                    'gender': gender
                })
                st.success("✅ Information submitted successfully!")
                # Optionally, navigate to Instructions tab
                # Not directly possible with tabs; consider prompting the user to switch tabs

# ---- Instructions Tab ----
with tabs[1]:
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

        3. **Atonal, Wide-Range Melody**
            - **Description:** A short sequence without a clear key, incorporating large jumps and diverse pitches.
            - **Reasoning:** Challenges the model to cope with irregular, disjunct patterns and to see if it imposes its own structure or explores outside tonal norms.

        4. **Long Repetitive Motif**
            - **Description:** An extended sequence made by repeating a two-bar motif several times.
            - **Reasoning:** Assesses how the model deals with longer context and repetitive patterns. Will it continue with variations, remain consistent, or diverge unexpectedly?
    """)

    if st.button("✅ Begin Testing"):
        # Optionally, automatically switch to the Testing tab or prompt user to do so
        st.success("You can now proceed to the Testing tab to begin your evaluation.")

# ---- Testing Tab ----
with tabs[2]:
    input_files = st.session_state.input_order
    current_input_index = st.session_state.current_input_index

    # Sidebar for navigation and progress
    with st.sidebar:
        st.header("🧭 Navigation")
        st.markdown("**Progress:**")
        current_step = 2 + (current_input_index * 4) + st.session_state.current_output_index  # Adjust as needed
        progress = current_step / st.session_state.total_steps
        st.progress(progress)
        st.markdown(f"Step {current_step} of {st.session_state.total_steps}")

        if st.button("🔄 Reset Session"):
            reset_session()

    if current_input_index < len(input_files):
        current_input_file = input_files[current_input_index]
        input_name = current_input_file.name  # e.g., 'input-3.mp4'
        descriptive_name = input_name_mapping.get(input_name, input_name)
        st.header(f"🔊 Listening to Input MIDI: {descriptive_name}")
        st.video(str(current_input_file))

        outputs = st.session_state.output_orders[current_input_file.name]
        output_index = st.session_state.current_output_index
        total_outputs = len(outputs)

        if output_index < total_outputs:
            current_output = outputs[output_index]
            continuation_number = output_index + 1  # 1 to 4
            output_file = current_output['file']
            st.subheader(f"🎹 Continuation {continuation_number} - Model: {current_output['model'].capitalize()}")
            st.video(str(output_file))

            with st.form(f"rating_form_{current_input_index}_{output_index}"):
                st.markdown("**Please rate the following criteria:**")
                for criterion in evaluation_criteria:
                    st.markdown(f"### {criterion['name']}")
                    st.markdown(f"*{criterion['description']}*")
                    st.markdown("**Scoring:**")
                    st.markdown(f"- **1:** {criterion['scoring']['1']}")
                    st.markdown(f"- **3:** {criterion['scoring']['3']}")
                    st.markdown(f"- **5:** {criterion['scoring']['5']}")
                    rating = st.slider(
                        f"Rate {criterion['name']}:",
                        min_value=1,
                        max_value=5,
                        value=3,
                        step=1,
                        key=f"{current_input_index}_{output_index}_{criterion['name']}"
                    )
                    # Save each criterion rating
                    st.session_state.responses.append({
                        'timestamp': datetime.now().isoformat(),
                        'page': 'testing',
                        'input': current_input_file.name,
                        'output': output_file.name,
                        'continuation_number': continuation_number,
                        'model': current_output['model'],
                        'criterion': criterion['name'],
                        'rating': rating
                    })
                    st.markdown("---")  # Divider between criteria

                submitted = st.form_submit_button("Submit Rating")
                if submitted:
                    st.success("✅ Rating submitted successfully!")
                    # Move to next output
                    st.session_state.current_output_index += 1
                    st.experimental_rerun()
        else:
            # Move to next input
            st.session_state.current_input_index += 1
            st.session_state.current_output_index = 0
            st.experimental_rerun()
    else:
        st.success("🎉 You have completed all the evaluations!")
        st.markdown("Please proceed to the **Closing** tab to submit your feedback.")

# ---- Closing Tab ----
with tabs[3]:
    st.image("closing_banner.png", use_container_width=True)
    st.title("✅ Thank You for Your Participation!")
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
            # Save all responses to the database
            save_ratings()
            save_feedback()
            st.success("✅ Your feedback has been submitted. Thank you!")
            st.stop()
