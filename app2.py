import streamlit as st
import os
import io
import random
import tempfile
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from mido import MidiFile
from datetime import datetime

# ======================================================
# ============== 1. APP CONFIG & STATE ================
# ======================================================
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

# ======================================================
# ============ 2. PIANO ROLL CODE (FROM 1ST) ==========
# ======================================================

# Minimal portions of the first script to parse and display a static piano roll
def parse_midi(file_path):
    """
    Parse a MIDI file and extract note information.
    Returns a pandas DataFrame with columns: note, velocity, start_time, duration.
    """
    midi = MidiFile(file_path)
    notes = []
    tempo = 500000  # Default tempo
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
                })
            elif msg.type in ('note_off',) or (msg.type == 'note_on' and msg.velocity == 0):
                for note_dict in reversed(notes):
                    if note_dict['note'] == msg.note and 'duration' not in note_dict:
                        note_end = track_times[i] * tempo / ticks_per_beat / 1_000_000
                        note_dict['duration'] = note_end - note_dict['start_time']
                        break

    # Remove notes without end times
    notes = [n for n in notes if 'duration' in n]

    # Adjust to start at 0
    if notes:
        min_start = min(n['start_time'] for n in notes)
        for n in notes:
            n['start_time'] -= min_start

    return pd.DataFrame(notes)

def generate_midi_note_names():
    """
    Generate a dictionary {note_number: note_name} for MIDI notes 21..108 (A0..C8).
    """
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']
    midi_note_names = {}
    for i in range(21, 109):
        octave = (i // 12) - 1
        note = note_names[i % 12]
        midi_note_names[i] = f"{note}{octave}"
    return midi_note_names

def display_piano_roll(df_notes):
    """
    Display a static piano roll (no moving red line).
    """
    if df_notes.empty:
        st.warning("No notes to display in the piano roll.")
        return

    # Create note name map
    midi_note_names = generate_midi_note_names()
    df_notes['note_name'] = df_notes['note'].apply(midi_note_names.get)

    # Filter out-of-range notes
    df_notes = df_notes.dropna(subset=['note_name'])

    # Identify unique notes actually used
    unique_notes = sorted(df_notes['note_name'].unique(),
                          key=lambda x: (
                              int(x.split('#')[0].replace('B','11').replace('A','9') if 'A' in x or 'B' in x else '0'), 
                              x
                          ))  # Not a perfect sorting, but we rely on the actual generation below
    # We'll do a simpler approach: just gather in ascending pitch order
    # Rebuild properly:
    # Sort by MIDI number instead:
    note_num_map = {nm: k for k, nm in generate_midi_note_names().items()}
    unique_notes = sorted(df_notes['note_name'].unique(), key=lambda x: note_num_map[x])

    note_to_y = {note: idx for idx, note in enumerate(unique_notes)}
    max_time = (df_notes['start_time'] + df_notes['duration']).max()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Piano Roll")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Notes")

    # Y-axis
    ax.set_yticks(range(len(unique_notes)))
    ax.set_yticklabels(unique_notes)

    # Plot each note as a bar
    for _, row in df_notes.iterrows():
        y = note_to_y[row['note_name']]
        start = row['start_time']
        duration = row['duration']
        ax.broken_barh([(start, duration)], (y - 0.4, 0.8),
                       facecolors='blue', edgecolors='black', linewidth=0.5)

    # X-axis limit
    ax.set_xlim(0, max_time + 1)
    ax.set_ylim(-1, len(unique_notes))

    ax.grid(True, linestyle='--', linewidth=0.5)

    st.pyplot(fig)

# ======================================================
# =========== 3. PAGES & MAIN NAVIGATION ==============
# ======================================================
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

# --------------------- WELCOME ------------------------
def welcome_page():
    st.title("🎵 Welcome to the AI Music Assistant 🎵")

    # Button before examples
    if st.button("Proceed to Examples"):
        st.session_state.page = 'instructions'
        st.rerun()

    st.markdown("Below are some **example** inputs and outputs from the user testing script.")

    # Show example videos with the same titles as in the user testing script
    # 1. Input 3
    st.video("output_videos/input-3.mp4")
    st.markdown("**Familiar Tonal Snippet (Input-3)**")

    # 2. Output 3 - Mono
    st.video("output_videos/output-3-mono.mp4")
    st.markdown("**Continuation (Model: Mono)**")

    # 3. Output 3 - Lookback
    st.video("output_videos/output-3-lookback.mp4")
    st.markdown("**Continuation (Model: Lookback)**")

# ------------------- INSTRUCTIONS ---------------------
def instructions_page():
    st.title("📋 Instructions")
    st.markdown("""
        - **Upload a MIDI file** or **select one of the sample MIDIs** on the next page.
        - **Choose a model** to produce a continuation (currently identical to your input).
        - **Listen** to your file, **view** its piano roll, and see how the system works.
    """)

    # A "fun fact" tucked away in an expander
    with st.expander("Fun Fact About This Project"):
        st.markdown("""
        This AI Music Assistant began as a university dissertation project on interactive deep learning
        for music. The goal is to eventually generate creative and coherent musical continuations.
        """)

    if st.button("Go to Model Selection"):
        st.session_state.page = 'select_model'
        st.rerun()

# -------------- SELECT MODEL & UPLOAD ---------------
def select_model_page():
    st.title("Select Model & Upload MIDI File")

    # Step-by-step guides or tooltips could be brief text hints
    st.markdown("""
        **Step 1:** Upload your MIDI file below.  
        **Step 2:** Or pick one from the dropdown of sample MIDIs if you don't have your own.  
        **Step 3:** Choose a model for a demonstration of how we generate continuations (currently returning your input).  
    """)

    # File Uploader
    file_uploader = st.file_uploader("Upload your MIDI file:", type=["mid"])

    # Dropdown for sample MIDIs
    sample_midis = {
        "Familiar Tonal Snippet (input-3)": "input_midis/input-3.mid",
        "Medium-Length Original Melody (input-4)": "input_midis/input-4.mid",
        "Atonal, Wide-Range Melody (input-5)": "input_midis/input-5.mid",
        "Long Repetitive Motif (input-6)": "input_midis/input-6.mid"
    }
    chosen_sample = st.selectbox("Or select a sample MIDI:", ["None"] + list(sample_midis.keys()))

    # Model descriptions
    model_descriptions = {
        'magenta melody rnn': "General RNN-based approach from the Magenta library.",
        'basic': "A simpler RNN model focusing on single-step continuity.",
        'lookback': "Incorporates historical context from further back in the sequence.",
        'attention': "Leverages attention layers to track relevant notes or motifs.",
        'mono': "Specially constrained to monophonic lines for clarity."
    }
    model_options = list(model_descriptions.keys())  # or a custom order
    selected_model = st.selectbox("Select a model:", model_options, index=0)

    # Interactive Tutorials or hints
    st.markdown("**Hint:** The 'Attention' model typically handles multi-layer sequences more gracefully.")

    # Confirm button
    if st.button("Load & Show Piano Roll"):
        # If user has uploaded a MIDI, use that. Otherwise, check sample choice.
        if file_uploader is not None:
            st.session_state.uploaded_midi = file_uploader.getvalue()
        else:
            if chosen_sample != "None":
                try:
                    path = sample_midis[chosen_sample]
                    with open(path, "rb") as f:
                        st.session_state.uploaded_midi = f.read()
                except Exception as e:
                    st.error(f"Error loading sample MIDI: {e}")
                    return
            else:
                st.error("Please upload a MIDI file or select a sample.")
                return
        st.session_state.selected_model = selected_model

        st.session_state.page = 'output'
        st.rerun()

# ---------------------- OUTPUT -----------------------
def output_page():
    st.title("Your MIDI & Model Selection")
    
    if not st.session_state.uploaded_midi:
        st.warning("No MIDI data available. Please go back and upload/select a file.")
        return

    # Show chosen model
    st.write(f"**Selected Model:** {st.session_state.selected_model}")

    # Save to a temporary file
    temp_path = "temp_user_midi.mid"
    with open(temp_path, "wb") as f:
        f.write(st.session_state.uploaded_midi)

    # Display piano roll
    try:
        df_notes = parse_midi(temp_path)
        if df_notes.empty:
            st.warning("No playable notes found in your MIDI file.")
        else:
            st.markdown("**Piano Roll Visualization:**")
            display_piano_roll(df_notes)
    except Exception as e:
        st.error(f"Error displaying piano roll: {e}")

    # Audio playback attempt (often doesn't work for MIDI in browsers)
    st.markdown("**Audio Playback (MIDI)**")
    try:
        # st.audio() for raw MIDI often doesn't play in many browsers,
        # but we'll demonstrate the call:
        st.audio(st.session_state.uploaded_midi, format="audio/midi")
        st.info("Note: Some browsers may not support direct MIDI playback.")
    except Exception as e:
        st.error(f"Audio player error: {e}")

    # Download button for the "generated" file (currently identical to input)
    st.download_button(
        label="Download Generated MIDI",
        data=st.session_state.uploaded_midi,
        file_name="generated_continuation.mid",
        mime="audio/midi"
    )

    # Button to go to closing page
    if st.button("Finish & Proceed"):
        st.session_state.page = 'closing'
        st.rerun()

# ---------------------- CLOSING ----------------------
def closing_page():
    # Show balloons only when arriving on this page, not on re-run button clicks
    st.title("✅ Thank You for Using the AI Music Assistant!")
    st.markdown("""
        You can try rerunning the same MIDI with a different model below.
        This page is where we conclude the session or test further models.
    """)
    st.balloons()

    # Rerun with different model
    st.subheader("Rerun with a Different Model")
    other_models = ['magenta melody rnn', 'basic', 'lookback', 'attention', 'mono']
    new_choice = st.selectbox("Select a new model:", other_models, key="re_model_select")

    if st.button("Rerun with Selected Model"):
        st.session_state.selected_model = new_choice
        # No balloons for re-run
        st.session_state.page = 'output'
        st.rerun()

# ======================================================
# ================ MAIN APP SEQUENCE ==================
# ======================================================
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
