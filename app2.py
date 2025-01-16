import streamlit as st
from sqlalchemy import create_engine, text
from pathlib import Path
import random
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="AI Music Assistant",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ================== Session State Initialisation ==================
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'

if 'responses' not in st.session_state:
    st.session_state.responses = []

# We'll store the uploaded MIDI file and the model selection here
if 'uploaded_midi' not in st.session_state:
    st.session_state.uploaded_midi = None

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None

# Function to reset the session
def reset_session():
    keys_to_remove = [
        'page', 'responses', 'uploaded_midi', 'selected_model', 
        'db_engine'
    ]
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]
    st.experimental_rerun()

# ================== Database Functions ==================
def init_db():
    """
    Initialise database connection.
    """
    if 'db_engine' not in st.session_state:
        try:
            db_url = (
                f"postgresql://{st.secrets['postgres']['user']}:"
                f"{st.secrets['postgres']['password']}@"
                f"{st.secrets['postgres']['host']}:"
                f"{st.secrets['postgres']['port']}/"
                f"{st.secrets['postgres']['database']}"
            )
            engine = create_engine(
                db_url, 
                connect_args={"sslmode": st.secrets['postgres']['sslmode']}
            )
            st.session_state.db_engine = engine
        except Exception as e:
            st.error(f"Error connecting to PostgreSQL: {e}")

def save_user_info():
    """
    Save user information to the database.
    """
    if 'db_engine' not in st.session_state:
        st.error("Database not initialised.")
        return
    engine = st.session_state.db_engine
    try:
        with st.spinner("Saving your information..."):
            with engine.begin() as connection:
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
        st.success("✅ Your information has been successfully saved. Thank you!")
    except Exception as e:
        st.error(f"Error saving user information: {e}")

def save_feedback():
    """
    Save user feedback to the database.
    """
    if 'db_engine' not in st.session_state:
        st.error("Database not initialised.")
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

# ================== Evaluation Criteria (Reused from Second Script) ==================
evaluation_criteria = [
    {
        "name": "Pitch/Tonal Coherence",
        "description": (
            "How well does the continuation fit into an implied key or tonal centre established by "
            "the primer melody (if any)? For atonal examples, does the model at least maintain consistent "
            "intervallic relationships or avoid overly jarring leaps without intent?"
        ),
        "scoring": {
            "1": "Completely random, no sense of pitch organisation",
            "3": "Generally consistent pitches, occasional off-key notes if tonal context applies",
            "5": "Strong tonal coherence, notes feel harmonically related or purposefully atonal but structured"
        }
    },
    {
        "name": "Rhythmic Stability & Flow",
        "description": (
            "Does the continuation maintain a clear rhythmic pulse and align well with the time signature "
            "and feel of the initial segment? Is it rhythmically logical, or too erratic and disjointed?"
        ),
        "scoring": {
            "1": "Erratic durations, awkward pauses, no clear rhythmic pattern",
            "3": "Some stability, but occasional unnatural note placements",
            "5": "Flows naturally, rhythmically coherent, and matches or complements the original rhythm"
        }
    },
    {
        "name": "Motivic & Thematic Continuity",
        "description": (
            "Does the continuation pick up on melodic motifs, patterns, or contours from the original melody "
            "and develop them further? Or does it abruptly shift to unrelated ideas?"
        ),
        "scoring": {
            "1": "Completely unrelated sequence, abrupt changes with no continuity",
            "3": "Some fragments of previous patterns appear, but not consistently",
            "5": "Smooth thematic development, clear recognition and transformation of initial motifs"
        }
    },
    {
        "name": "Range & Complexity Management",
        "description": (
            "How well does the model manage pitch range and complexity? Does it stay within a logical register, "
            "or jump erratically across octaves? Is the complexity (interval sizes, rhythmic subdivisions) "
            "appropriate for the style?"
        ),
        "scoring": {
            "1": "Extreme register jumps without reason, overly complex or simplistic to the point of incoherence",
            "3": "Generally appropriate complexity, a few overly large leaps or simplistic streaks",
            "5": "Balanced complexity, maintains a comfortable range and adds variety without chaos"
        }
    },
    {
        "name": "Aesthetic/Subjective Appeal",
        "description": (
            "A more subjective measure of how pleasant, engaging, or musically “satisfying” the continuation sounds."
        ),
        "scoring": {
            "1": "Unpleasant, jarring, or meaningless sequence",
            "3": "Acceptable but somewhat bland or directionless",
            "5": "Musically appealing, has a sense of purpose or emotional quality"
        }
    }
]

# ================== Sidebar Renderer ==================
def render_sidebar():
    st.sidebar.title("📋 Contents")
    st.sidebar.markdown("---")
    
    # Navigation pages in the order we'll use them
    pages = ["welcome", "instructions", "select_model", "output", "closing"]
    for page in pages:
        # Capitalise the page names for display
        display_name = page.replace("_", " ").capitalize()
        if st.session_state.page == page:
            st.sidebar.markdown(f"### **{display_name}**")
        else:
            st.sidebar.markdown(display_name)
    
    st.sidebar.markdown("---")

    if st.sidebar.button("🔄 Reset Session"):
        reset_session()

# ================== Page 1: Welcome ==================
def welcome_page():
    st.image("banner.png", use_container_width=True)  # If you have a banner image
    st.title("🎵 Welcome to the AI Music Assistant 🎵")
    st.markdown("""
        Thank you for trying out this AI Music Assistant. In this application, you'll be able to upload 
        a MIDI file, choose a model to generate a continuation, and evaluate or download the output. 
        Your feedback will help us enhance this system's musical capabilities.
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
        submitted = st.form_submit_button("🚀 Continue")
        if submitted:
            # Validate inputs
            if not age.isdigit():
                st.error("Please enter a valid age.")
            elif musical_background == "Select":
                st.error("Please select your musical background.")
            elif gender == "Select":
                st.error("Please select your gender.")
            else:
                # Save user info in session state
                st.session_state.responses.append({
                    'timestamp': datetime.now().isoformat(),
                    'page': 'welcome',
                    'musical_background': musical_background,
                    'age': age,
                    'gender': gender
                })
                # Immediately attempt to save to DB if required
                save_user_info()
                st.session_state.page = 'instructions'
                st.experimental_rerun()

# ================== Page 2: Instructions ==================
def instructions_page():
    st.title("📋 Instructions")
    st.markdown("""
        ### Scoring Method

        - For each model’s generated continuation of your uploaded melody, you can assign a score of **1 to 5** for each criterion.
        - **Total the scores** for a composite result, or consider different weights for each criterion depending on priorities.
        - **Compare results** across different model selections or parameter settings to draw conclusions about which configurations yield the most coherent, appealing, and stylistically appropriate continuations.

        ### Approach

        1. Upload your MIDI file on the next page.
        2. Select a model from the available list.
        3. The system will (for now) simply return your exact MIDI file as a “continuation.” 
           Future versions will generate genuinely new continuations.
        4. You may listen to or download the output, view a piano roll video, and optionally provide a rating or feedback.

        When you are ready, click below to proceed.
    """)
    if st.button("✅ Proceed to Model Selection"):
        st.session_state.page = 'select_model'
        st.experimental_rerun()

# ================== Page 3: Select Model & Upload ==================
def select_model_page():
    st.title("🎶 Select Model & Upload MIDI File")
    st.markdown("""
        Upload your MIDI file below and choose which model you would like to use for generating a continuation.
        Currently, the output will simply mirror your input MIDI while the generation feature is under development.
    """)

    # Provide a file uploader for MIDI
    uploaded_file = st.file_uploader("Please upload a MIDI file (with .mid extension):", type=["mid"])

    # Let the user pick from a set of models, same as second script
    models = ['attention', 'basic', 'lookback', 'mono']
    chosen_model = st.selectbox("Choose a model:", models)

    if st.button("Generate Continuation"):
        if uploaded_file is None:
            st.error("Please upload a MIDI file first.")
        else:
            st.session_state.uploaded_midi = uploaded_file
            st.session_state.selected_model = chosen_model
            # For now, do nothing advanced, just proceed to the 'output' page
            st.session_state.page = 'output'
            st.experimental_rerun()

# ================== Page 4: Output & Rating ==================
def output_page():
    st.title("🎼 Output Continuation")
    st.markdown("""
        Below is the continuation generated by the selected model. 
        At this stage, we are simply returning the same file you uploaded. 
        Future versions will provide genuine AI-driven continuations.
    """)

    if st.session_state.uploaded_midi is None:
        st.warning("No MIDI file found. Please go back and upload a MIDI file.")
        return

    st.write(f"**Selected Model:** `{st.session_state.selected_model}`")
    # Displaying or 'playing' the MIDI in Streamlit is not straightforward,
    # but we can provide a download link and instructions.

    # We'll store the uploaded file in memory. Let's create a download button for the user's MIDI.
    st.download_button(
        label="Download Generated MIDI",
        data=st.session_state.uploaded_midi.getvalue(),
        file_name="generated_continuation.mid",
        mime="audio/midi"
    )

    st.markdown("---")

    # ========== Optional Ratings - Reusing Criteria from the second script ==========
    st.subheader("Rate the Generated Continuation (Optional)")
    st.markdown("""
        Although this is currently just your own file, please feel free to experiment 
        with the rating criteria below. This will illustrate how the system collects 
        and stores feedback for future AI-generated continuations.
    """)

    # Create form to collect ratings
    with st.form("rating_form"):
        st.markdown("**Please rate the following criteria:**")

        tabs = st.tabs([criterion['name'] for criterion in evaluation_criteria])
        ratings = {}
        unanswered_criteria = []

        for tab, criterion in zip(tabs, evaluation_criteria):
            with tab:
                st.markdown(f"*{criterion['description']}*")
                st.markdown("**Scoring:**")
                st.markdown(f"- **1:** {criterion['scoring']['1']}")
                st.markdown(f"- **3:** {criterion['scoring']['3']}")
                st.markdown(f"- **5:** {criterion['scoring']['5']}")
                
                rating = st.selectbox(
                    f"Rate {criterion['name']}:",
                    options=['Select', 1, 2, 3, 4, 5],
                    index=0,
                    key=f"rating_{criterion['name']}"
                )
                ratings[criterion['name']] = rating

        submitted = st.form_submit_button("Submit Ratings")
        if submitted:
            for criterion_name, rating in ratings.items():
                if rating == 'Select':
                    unanswered_criteria.append(criterion_name)
            if unanswered_criteria:
                st.error("Please rate all criteria before submitting.")
                st.warning(f"Unanswered Criteria: {', '.join(unanswered_criteria)}")
            else:
                # Append all ratings to session state
                for criterion_name, rating_value in ratings.items():
                    st.session_state.responses.append({
                        'timestamp': datetime.now().isoformat(),
                        'page': 'output',  # You might rename or keep for reference
                        'model': st.session_state.selected_model,
                        'criterion': criterion_name,
                        'rating': rating_value
                    })
                st.success("✅ Ratings submitted successfully!")

    st.markdown("---")

    # Button to proceed to Closing
    if st.button("Finish & Proceed"):
        st.session_state.page = 'closing'
        st.experimental_rerun()

# ================== Page 5: Closing ==================
def closing_page():
    st.image("closing_banner.png", use_container_width=True)  # If you have a closing banner
    st.title("✅ Thank You for Using the AI Music Assistant!")
    st.markdown("""
        We appreciate you taking the time to explore this assistant. 
        Your feedback is invaluable and will contribute to the development of more sophisticated musical tools.
    """)

    st.balloons()  # Celebration effect

    with st.form("additional_feedback"):
        feedback = st.text_area(
            "📝 Do you have any additional comments or suggestions?", 
            ""
        )
        submitted = st.form_submit_button("📤 Submit Feedback & Exit")
        if submitted:
            if feedback:
                st.session_state.responses.append({
                    'timestamp': datetime.now().isoformat(),
                    'page': 'feedback',
                    'feedback': feedback
                })
            # Save the feedback to the database
            save_feedback()
            st.markdown("### You may now close this window. Thanks again!")
            st.stop()

# ================== Main Script Flow ==================
init_db()         # Initialise the DB connection if possible
render_sidebar()  # Render the sidebar each time

# Navigation through pages
if st.session_state.page == 'welcome':
    welcome_page()
elif st.session_state.page == 'instructions':
    instructions_page()
elif st.session_state.page == 'select_model':
    select_model_page()
elif st.session_state.page == 'output':
    output_page()
elif st.session_state.page == 'closing':
    closing_page()
else:
    st.error("Unknown page!")
