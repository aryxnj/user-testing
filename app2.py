import streamlit as st
from pathlib import Path
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

if 'uploaded_midi' not in st.session_state:
    st.session_state.uploaded_midi = None

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None

# Function to reset the session
def reset_session():
    keys_to_remove = [
        'page', 'responses', 'uploaded_midi', 'selected_model'
    ]
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# ================== Evaluation Criteria ==================
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
    
    pages = ["welcome", "instructions", "select_model", "output", "closing"]
    for page in pages:
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
    st.image("banner.png", use_container_width=True)
    st.title("🎵 Welcome to the AI Music Assistant 🎵")
    st.markdown("""
        Thank you for trying out this AI Music Assistant. In this application, you'll be able to upload 
        a MIDI file, choose a model to generate a continuation, and evaluate or download the output. 
        Your feedback will help us enhance this system's musical capabilities.
    """)
    if st.button("🚀 Continue"):
        st.session_state.page = 'instructions'
        st.rerun()

# ================== Page 2: Instructions ==================
def instructions_page():
    st.title("📋 Instructions")
    st.markdown("""
        ### Approach

        1. Upload your MIDI file on the next page.
        2. Select a model from the available list.
        3. The system will (for now) simply return your exact MIDI file as a “continuation.” 
           Future versions will generate genuinely new continuations.
        4. You may listen to or download the output, view a piano roll video, and optionally provide feedback.

        When you are ready, click below to proceed.
    """)
    if st.button("✅ Proceed to Model Selection"):
        st.session_state.page = 'select_model'
        st.rerun()

# ================== Page 3: Select Model & Upload ==================
def select_model_page():
    st.title("🎶 Select Model & Upload MIDI File")
    st.markdown("""
        Upload your MIDI file below and choose which model you would like to use for generating a continuation.
        Currently, the output will simply mirror your input MIDI while the generation feature is under development.
    """)

    uploaded_file = st.file_uploader("Please upload a MIDI file (with .mid extension):", type=["mid"])
    models = ['attention', 'basic', 'lookback', 'mono']
    chosen_model = st.selectbox("Choose a model:", models)

    if st.button("Generate Continuation"):
        if uploaded_file is None:
            st.error("Please upload a MIDI file first.")
        else:
            st.session_state.uploaded_midi = uploaded_file
            st.session_state.selected_model = chosen_model
            st.session_state.page = 'output'
            st.rerun()

# ================== Page 4: Output ==================
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

    st.download_button(
        label="Download Generated MIDI",
        data=st.session_state.uploaded_midi.getvalue(),
        file_name="generated_continuation.mid",
        mime="audio/midi"
    )

    st.markdown("---")

    if st.button("Finish & Proceed"):
        st.session_state.page = 'closing'
        st.rerun()

# ================== Page 5: Closing ==================
def closing_page():
    st.image("closing_banner.png", use_container_width=True)
    st.title("✅ Thank You for Using the AI Music Assistant!")
    st.markdown("""
        We appreciate you taking the time to explore this assistant. 
        Your feedback will contribute to the development of more sophisticated musical tools.
    """)
    st.balloons()

    st.markdown("---")
    st.subheader("🔄 Rerun with a Different Model")
    st.markdown("Select a different model to rerun the continuation with the same MIDI file.")

    models = ['attention', 'basic', 'lookback', 'mono']
    new_model = st.selectbox("Choose a new model:", models, key="new_model_select")

    if st.button("Run with New Model"):
        st.session_state.selected_model = new_model
        st.session_state.page = 'output'
        st.rerun()

# ================== Main Script Flow ==================
render_sidebar()

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
