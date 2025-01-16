import streamlit as st
from pathlib import Path
from datetime import datetime
import io
import matplotlib.pyplot as plt
from mido import MidiFile

# ================== Utility Functions from First Script ==================
def parse_midi(file_path):
    """Parse a MIDI file and extract note information."""
    midi = MidiFile(file_path)
    notes = []
    tempo = 500000
    ticks_per_beat = midi.ticks_per_beat
    track_times = [0] * len(midi.tracks)

    for i, track in enumerate(midi.tracks):
        for msg in track:
            track_times[i] += msg.time
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            elif msg.type == 'note_on' and msg.velocity > 0:
                note_start = track_times[i] * tempo / ticks_per_beat / 1_000_000
                notes.append({
                    'note': msg.note,
                    'velocity': msg.velocity,
                    'start_time': note_start,
                    'duration': None
                })
            elif (msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)):
                for note in reversed(notes):
                    if note['note'] == msg.note and note['duration'] is None:
                        note_end = track_times[i] * tempo / ticks_per_beat / 1_000_000
                        note['duration'] = note_end - note['start_time']
                        break
    notes = [n for n in notes if n['duration'] is not None]
    return notes

def generate_midi_note_names():
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    midi_note_names = {}
    for i in range(21, 109):
        octave = (i // 12) - 1
        note = note_names[i % 12]
        midi_note_names[i] = f"{note}{octave}"
    return midi_note_names

def plot_piano_roll(notes):
    midi_note_names = generate_midi_note_names()
    unique_notes = sorted({midi_note_names[n['note']] for n in notes if n['note'] in midi_note_names})
    note_to_y = {note: idx for idx, note in enumerate(unique_notes)}

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_ylim(-1, len(unique_notes))
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Notes")
    ax.set_yticks(range(len(unique_notes)))
    ax.set_yticklabels(unique_notes)
    ax.grid(True, linestyle='--', linewidth=0.5)

    for note_info in notes:
        note_number = note_info['note']
        note_name = midi_note_names.get(note_number)
        if note_name and note_name in note_to_y:
            y = note_to_y[note_name]
            start = note_info['start_time']
            duration = note_info['duration']
            ax.broken_barh([(start, duration)], (y - 0.4, 0.8), facecolors='blue', alpha=0.5)

    st.pyplot(fig)

# ================== Streamlit App Setup ==================
st.set_page_config(
    page_title="AI Music Assistant",
    layout="centered",
    initial_sidebar_state="expanded"
)

if 'page' not in st.session_state:
    st.session_state.page = 'welcome'
if 'uploaded_midi' not in st.session_state:
    st.session_state.uploaded_midi = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None

def reset_session():
    for key in ['page', 'uploaded_midi', 'selected_model']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

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
        Thank you for trying out this AI Music Assistant. You can upload a MIDI file, select a model,
        and generate a continuation. Enjoy exploring!
    """)
    if st.button("🚀 Continue"):
        st.session_state.page = 'instructions'
        st.rerun()
    st.video("input_midis/input-3.mp4")
    st.markdown("**Example Input 3: Familiar Tonal Snippet**")
    st.video("output_videos/output-3-mono.mp4")
    st.markdown("**Example Output 3 - Mono Model**")
    st.video("output_videos/output-3-lookback.mp4")
    st.markdown("**Example Output 3 - Lookback Model**")

# ================== Page 2: Instructions ==================
def instructions_page():
    st.title("📋 Instructions")
    st.markdown("""
        **Approach:**
        1. Upload a MIDI file and select a model.
        2. The system will return your MIDI file as a “continuation.”
        3. Listen, download, and view dynamic visualizations of the output.
        4. Use the provided tips and tutorials to guide you.
    """)
    # Fun Fact Pop-up
    if st.button("Show Fun Fact"):
        st.info("🤖 **Fun Fact:** Our AI Music Assistant is trained on thousands of MIDI files to understand musical patterns!")
    if st.button("✅ Proceed to Model Selection"):
        st.session_state.page = 'select_model'
        st.rerun()

# ================== Page 3: Select Model & Upload ==================
def select_model_page():
    st.title("🎶 Select Model & Upload MIDI File")
    st.markdown("""
        Upload your MIDI file and choose a model. If you don't have a file, select a sample from the dropdown.
    """)

    model_descriptions = {
        'attention': "Uses attention mechanisms to focus on relevant musical patterns.",
        'basic': "A simple RNN-based model for melody continuation.",
        'lookback': "Considers previous sections of music to generate coherent continuations.",
        'mono': "Generates monophonic melodies with basic structure."
    }
    models = ['attention', 'basic', 'lookback', 'mono']
    st.markdown("**Model Options:**")
    for model in models:
        st.markdown(f"- **{model.capitalize()}**: {model_descriptions[model]}")

    uploaded_file = st.file_uploader("Upload a MIDI file (.mid):", type=["mid"], help="Click to upload your own MIDI file.")
    chosen_model = st.selectbox("Choose a model:", models, help="Select a model to generate the continuation.")

    sample_files = {
        "Familiar Tonal Snippet": "input_midis/input-3.mid",
        "Medium-Length Original Melody": "input_midis/input-4.mid",
        "Atonal, Wide-Range Melody": "input_midis/input-5.mid",
        "Long Repetitive Motif": "input_midis/input-6.mid"
    }
    selected_sample = st.selectbox("Or select a sample MIDI file:", list(sample_files.keys()))

    # Load sample if chosen
    if selected_sample:
        try:
            with open(sample_files[selected_sample], "rb") as f:
                st.session_state.uploaded_midi = io.BytesIO(f.read())
            st.info(f"Loaded sample: {selected_sample}")
        except Exception as e:
            st.error(f"Error loading sample: {e}")

    if st.button("Generate Continuation"):
        if st.session_state.uploaded_midi is None and uploaded_file is None:
            st.error("Please upload a MIDI file or select a sample first.")
        else:
            if uploaded_file is not None:
                st.session_state.uploaded_midi = uploaded_file
            st.session_state.selected_model = chosen_model
            st.session_state.page = 'output'
            st.rerun()

# ================== Page 4: Output ==================
def output_page():
    st.title("🎼 Output Continuation")
    st.markdown("""
        Below is the continuation generated by the selected model. Currently, it's the same as your input.
    """)

    if st.session_state.uploaded_midi is None:
        st.warning("No MIDI file found. Please go back and upload a MIDI file.")
        return

    st.write(f"**Selected Model:** `{st.session_state.selected_model}`")

    # Show piano roll immediately
    try:
        uploaded_midi_data = st.session_state.uploaded_midi.getvalue()
        with open("temp_uploaded.mid", "wb") as temp_file:
            temp_file.write(uploaded_midi_data)
        notes = parse_midi("temp_uploaded.mid")
        plot_piano_roll(notes)
    except Exception as e:
        st.error(f"Error generating piano roll: {e}")

    st.markdown("### Listen to Your MIDI")
    st.audio(st.session_state.uploaded_midi.getvalue(), format="audio/midi")

    st.download_button(
        label="Download Generated MIDI",
        data=st.session_state.uploaded_midi.getvalue(),
        file_name="generated_continuation.mid",
        mime="audio/midi"
    )

    st.markdown("---")
    # Button to generate piano roll video (placeholder)
    if st.button("Generate Piano Roll Video"):
        st.info("Piano roll video generation initiated. (Feature under development)")

    if st.button("Finish & Proceed"):
        st.session_state.page = 'closing'
        st.rerun()

# ================== Page 5: Closing ==================
def closing_page():
    st.image("closing_banner.png", use_container_width=True)
    st.title("✅ Thank You for Using the AI Music Assistant!")
    st.markdown("""
        We appreciate you exploring the AI Music Assistant. You can try running the same MIDI file with a different model below.
    """)
    # Trigger balloons only on initial page load
    if 'balloons_shown' not in st.session_state:
        st.balloons()
        st.session_state.balloons_shown = True

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
