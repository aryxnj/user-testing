import streamlit as st
from pathlib import Path
import os
import io
import random
import matplotlib.pyplot as plt
import pandas as pd
from mido import MidiFile

# -------------- PIANO ROLL GENERATION UTILITIES (Simplified) --------------
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
                    'start_time': note_start
                })
            elif msg.type in ['note_off', 'note_on'] and msg.velocity == 0:
                # Find matching note that doesn't have a duration yet
                for note in reversed(notes):
                    if note['note'] == msg.note and 'duration' not in note:
                        note_end = track_times[i] * tempo / ticks_per_beat / 1_000_000
                        note['duration'] = note_end - note['start_time']
                        break

    # Remove notes without duration
    notes = [n for n in notes if 'duration' in n]

    # Shift start times so the earliest note begins at 0
    if notes:
        min_start = min(n['start_time'] for n in notes)
        for n in notes:
            n['start_time'] -= min_start

    return pd.DataFrame(notes)


def generate_midi_note_names():
    """
    Generate a mapping from MIDI note numbers to note names.
    Returns a dictionary {note_number: note_name}.
    """
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']
    midi_note_names = {}
    for i in range(21, 109):  # A0 (21) to C8 (108)
        octave = (i // 12) - 1
        note = note_names[i % 12]
        midi_note_names[i] = f"{note}{octave}"
    return midi_note_names


def plot_piano_roll(df_notes):
    """
    Plots a piano roll (without the red time line).
    """
    if df_notes.empty:
        st.warning("No notes to plot.")
        return

    midi_note_names = generate_midi_note_names()
    # Convert numeric note -> name, but remove anything out of range
    df_notes['note_name'] = df_notes['note'].map(midi_note_names).dropna()

    # Filter out-of-range notes if they exist
    valid_notes = df_notes.dropna(subset=['note_name']).copy()
    if valid_notes.empty:
        st.warning("All notes are out of piano range. Nothing to display.")
        return

    # Create a set of all note names used
    used_note_names = sorted(valid_notes['note_name'].unique())

    # Map note_name -> y-axis index
    note_to_y = {name: i for i, name in enumerate(used_note_names)}
    max_time = (valid_notes['start_time'] + valid_notes['duration']).max()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_ylim(-1, len(used_note_names))
    ax.set_xlim(0, max_time * 1.05)  # a bit of padding
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Notes")
    ax.set_title("Piano Roll")
    ax.set_yticks(range(len(used_note_names)))
    ax.set_yticklabels(used_note_names)
    ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)

    for _, row in valid_notes.iterrows():
        start = row['start_time']
        duration = row['duration']
        y = note_to_y[row['note_name']]
        ax.broken_barh([(start, duration)], (y - 0.4, 0.8),
                       facecolors='blue', edgecolors='black', alpha=0.6)

    st.pyplot(fig)


# -------------- STREAMLIT APP --------------
st.set_page_config(page_title="AI Music Assistant", layout="centered", initial_sidebar_state="expanded")

# Session State Setup
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'
if 'uploaded_midi' not in st.session_state:
    st.session_state.uploaded_midi = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'sample_file' not in st.session_state:
    st.session_state.sample_file = None

def reset_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# ---- Sidebar ----
def render_sidebar():
    st.sidebar.title("Contents")
    st.sidebar.markdown("---")
    pages = ["welcome", "instructions", "select_model", "output", "closing"]
    for p in pages:
        disp_name = p.replace("_", " ").capitalize()
        if st.session_state.page == p:
            st.sidebar.markdown(f"### **{disp_name}**")
        else:
            st.sidebar.markdown(disp_name)
    st.sidebar.markdown("---")

    if st.sidebar.button("Reset Session"):
        reset_session()

# ---- Pages ----
def welcome_page():
    st.title("🎵 Welcome to the AI Music Assistant 🎵")
    st.markdown("Please click the button below to begin.")
    
    # Button before examples
    if st.button("Begin →"):
        st.session_state.page = "instructions"
        st.rerun()

    # Title and display example videos (like in user testing script)
    st.markdown("### Example: Familiar Tonal Snippet")
    st.video("output_videos/input-3.mp4")
    st.markdown("**Model Output: Mono**")
    st.video("output_videos/output-3-mono.mp4")
    st.markdown("**Model Output: Lookback**")
    st.video("output_videos/output-3-lookback.mp4")

def instructions_page():
    st.title("Instructions")
    st.markdown("""
        1. Upload or select a sample MIDI file.
        2. Choose a model to generate a continuation (for now, the same MIDI).
        3. View a quick piano roll, listen to the file, and optionally create a video continuation.
    """)

    # A "fun fact" about this project (tucked away)
    with st.expander("ℹ️ Fun Fact about this Project"):
        st.write("This project uses deep learning models originally designed by Magenta to analyse and generate melodic ideas in a variety of styles.")

    if st.button("Proceed to Model Selection"):
        st.session_state.page = "select_model"
        st.rerun()

def select_model_page():
    st.title("Select Model & Upload MIDI File")

    st.markdown("**Models:**")
    model_descriptions = {
        'magenta_melody_rnn_basic': "A basic RNN-based approach for melody continuation.",
        'magenta_melody_rnn_lookback': "A lookback model that references prior steps in the sequence.",
        'magenta_melody_rnn_attention': "An attention-based model that learns to focus on key notes/timings.",
        'magenta_melody_rnn_mono': "A monophonic RNN model for generating single-voice melodies."
    }
    model_options = list(model_descriptions.keys())

    st.session_state.selected_model = st.selectbox(
        "Choose a Model:",
        model_options,
        help="Select the model you'd like to use for generating a continuation."
    )
    for model_name in model_options:
        if model_name == st.session_state.selected_model:
            st.info(f"**{model_name}:** {model_descriptions[model_name]}")

    # Upload or Sample File
    uploaded_file = st.file_uploader("Upload a MIDI File (.mid):", type=["mid"], help="Click to upload your own MIDI file.")
    
    # Use sample file from a dropdown
    st.markdown("**Or select a sample MIDI file below:**")
    sample_options = {
        "input-3.mid": "Familiar Tonal Snippet",
        "input-4.mid": "Medium-Length Original Melody",
        "input-5.mid": "Atonal, Wide-Range Melody",
        "input-6.mid": "Long Repetitive Motif"
    }
    sample_choice = st.selectbox("Choose a sample:", ["None"] + list(sample_options.keys()))

    if st.button("Load Sample"):
        if sample_choice != "None":
            try:
                # We'll store the chosen sample path in 'input_midis/' directory
                path = Path("input_midis") / sample_choice
                with open(path, "rb") as f:
                    st.session_state.uploaded_midi = io.BytesIO(f.read())
                st.session_state.sample_file = sample_choice
                st.info(f"Sample loaded: {sample_options[sample_choice]}")
            except Exception as e:
                st.error(f"Error loading sample: {e}")

    # If the user pressed "Confirm & Show"
    if st.button("Confirm & Show"):
        # Priority: if user uploaded a file, override sample
        if uploaded_file is not None:
            st.session_state.uploaded_midi = uploaded_file
            st.session_state.sample_file = None

        if st.session_state.uploaded_midi is None:
            st.error("No MIDI selected or uploaded.")
        else:
            st.session_state.page = "output"
            st.rerun()

def output_page():
    st.title("Generated Continuation")
    st.markdown("Below is your chosen or uploaded MIDI. Currently, the continuation is simply the same file.")

    if st.session_state.uploaded_midi is None:
        st.warning("No MIDI found. Please return to model selection.")
        return

    st.write(f"**Selected Model:** {st.session_state.selected_model}")

    # Let user play or download the file
    midi_data = st.session_state.uploaded_midi.getvalue()
    st.download_button(
        label="Download MIDI",
        data=midi_data,
        file_name="generated_continuation.mid",
        mime="audio/midi"
    )

    # Show Piano Roll immediately
    try:
        # Write the file to a temp location to parse
        with open("temp_uploaded.mid", "wb") as f:
            f.write(midi_data)
        df_notes = parse_midi("temp_uploaded.mid")
        plot_piano_roll(df_notes)
    except Exception as e:
        st.error(f"Error generating piano roll: {e}")

    # Attempt to embed an audio player or waveform: 
    # (Note: Many browsers won't natively play .mid files. You may need a custom MIDI player or a .wav fallback.)
    st.markdown("### Listen to the MIDI")
    st.markdown("*(Note: Native browser playback of .mid files may not be supported in all browsers.)*")
    st.audio(midi_data, format="audio/midi")

    # Button to create a piano roll video (using first script) - placeholder
    st.markdown("---")
    if st.button("Generate Piano Roll Video"):
        st.info("Video generation (placeholder) - would call the script to produce the .mp4 and provide download link.")

    if st.button("Finish & Proceed"):
        st.session_state.page = "closing"
        st.rerun()

def closing_page():
    st.title("Thank You")
    st.markdown("Your exploration helps refine our AI Music Assistant. Feel free to select another model below or simply close this page.")
    
    # We only show balloons ONCE, upon arriving on this page:
    if 'closing_balloons_shown' not in st.session_state:
        st.session_state.closing_balloons_shown = True
        st.balloons()

    st.markdown("---")
    st.subheader("Rerun with a Different Model")
    new_model = st.selectbox("Choose a new model:", [
        'magenta_melody_rnn_basic',
        'magenta_melody_rnn_lookback',
        'magenta_melody_rnn_attention',
        'magenta_melody_rnn_mono'
    ], key="new_model_select")
    if st.button("Re-run with New Model"):
        st.session_state.selected_model = new_model
        # Do NOT show balloons here. We'll jump to output directly:
        st.session_state.page = "output"
        st.rerun()

# ---- Main Flow ----
render_sidebar()
if st.session_state.page == "welcome":
    welcome_page()
elif st.session_state.page == "instructions":
    instructions_page()
elif st.session_state.page == "select_model":
    select_model_page()
elif st.session_state.page == "output":
    output_page()
elif st.session_state.page == "closing":
    closing_page()
else:
    st.error("Unknown page!")
