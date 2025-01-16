import streamlit as st
import os
import io
import tempfile
import shutil
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from mido import MidiFile
from midi2audio import FluidSynth
from tqdm import tqdm

# =======================================
# ====== Copy of First Script Bits ======
# =======================================

# Adjust your SoundFont path as appropriate
SOUND_FONT_PATH = os.path.join(os.getcwd(), 'sounds', 'FluidR3_GM.sf2')

# You could adjust these if desired
FPS = 30
BPM = 130
BEATS_PER_BAR = 4
SECONDS_PER_BEAT = 60 / BPM
SECONDS_PER_BAR = SECONDS_PER_BEAT * BEATS_PER_BAR

fs = FluidSynth(sound_font=SOUND_FONT_PATH)

def parse_midi(file_path: str) -> pd.DataFrame:
    """
    Parse a MIDI file and extract note information.
    Returns a DataFrame with columns: note, velocity, start_time, duration.
    """
    midi = MidiFile(file_path)
    notes = []
    tempo = 500000  # default tempo
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
            elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
                # Match the last note_on (reversed for correct pairing)
                for note_data in reversed(notes):
                    if note_data['note'] == msg.note and 'duration' not in note_data:
                        note_end = track_times[i] * tempo / ticks_per_beat / 1_000_000
                        note_data['duration'] = note_end - note_data['start_time']
                        break

    # Remove notes that never received a duration
    notes = [n for n in notes if 'duration' in n]

    # Align all start times so that the earliest note begins at 0
    if notes:
        min_start = min(n['start_time'] for n in notes)
        for n in notes:
            n['start_time'] -= min_start

    return pd.DataFrame(notes)

def generate_midi_note_names() -> dict:
    """
    Generate a mapping from MIDI note numbers to note names.
    """
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    midi_note_names = {}
    for i in range(21, 109):
        octave = (i // 12) - 1
        note = note_names[i % 12]
        midi_note_names[i] = f"{note}{octave}"
    return midi_note_names

def create_piano_roll(data: pd.DataFrame):
    """
    Create a static piano roll plot (no red line). Returns a matplotlib figure.
    """
    midi_note_names = generate_midi_note_names()
    # Filter for standard piano range
    data['note_name'] = data['note'].map(midi_note_names).dropna()
    valid_data = data[data['note_name'].notnull()].copy()

    # Identify unique notes actually used
    unique_notes_used = sorted(valid_data['note_name'].unique(),
                               key=lambda x: (int(x[-1]), x[:-1]))  # Sort by octave then pitch

    if not len(valid_data):
        # Return an empty figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No notes to display.", ha='center', va='center', fontsize=14)
        return fig

    # Create mappings
    note_to_y = {note: idx for idx, note in enumerate(unique_notes_used)}

    # Plot
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_yticks(range(len(unique_notes_used)))
    ax.set_yticklabels(unique_notes_used)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Notes")
    ax.set_title("Simple Piano Roll")
    ax.grid(True, linestyle='--', linewidth=0.5)

    for _, row in valid_data.iterrows():
        note_name = row['note_name']
        y = note_to_y[note_name]
        start = row['start_time']
        duration = row['duration']
        ax.broken_barh([(start, duration)], (y - 0.4, 0.8), facecolors='blue')

    plt.tight_layout()
    return fig

def convert_midi_to_wav(midi_data: bytes) -> bytes:
    """
    Convert MIDI data (as bytes) to WAV (as bytes) using FluidSynth.
    Returns WAV file bytes.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp_midi:
        tmp_midi.write(midi_data)
        tmp_midi_path = tmp_midi.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        tmp_wav_path = tmp_wav.name

    fs.midi_to_audio(tmp_midi_path, tmp_wav_path)

    with open(tmp_wav_path, 'rb') as f:
        wav_bytes = f.read()

    # Cleanup
    os.remove(tmp_midi_path)
    os.remove(tmp_wav_path)

    return wav_bytes

def generate_piano_roll_video(midi_bytes: bytes) -> bytes:
    """
    Example placeholder: generate a piano roll video from midi_bytes using the first script's logic.
    For this example, we'll skip the actual FFmpeg steps and just return b'VIDEO' as a placeholder.
    """
    # In a real scenario, you'd replicate the logic from create_video_frames -> create_video_ffmpeg
    return b"VIDEO_PLACEHOLDER"

# ===================================
# ====== Streamlit App Starts =======
# ===================================

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
if 'sample_midi_data' not in st.session_state:
    st.session_state.sample_midi_data = None

def reset_session():
    for key in ['page', 'uploaded_midi', 'selected_model', 'sample_midi_data']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

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
    st.title("🎵 Welcome to the AI Music Assistant 🎵")

    # Button before the example videos
    if st.button("Continue to Instructions"):
        st.session_state.page = 'instructions'
        st.rerun()

    # Title the example videos (taken from the user testing script style)
    st.markdown("## Example Input Video: Familiar Tonal Snippet (input-3)")
    st.video("output_videos/input-3.mp4")

    st.markdown("## Example Output Video: Continuation by Mono Model (output-3-mono)")
    st.video("output_videos/output-3-mono.mp4")

    st.markdown("## Example Output Video: Continuation by Lookback Model (output-3-lookback)")
    st.video("output_videos/output-3-lookback.mp4")

# ================== Page 2: Instructions ==================
def instructions_page():
    st.title("Instructions")
    st.markdown("1. Upload a MIDI file or select a sample.\n"
                "2. Choose a model.\n"
                "3. Observe the piano roll and listen to your MIDI.\n"
                "4. Download outputs if you wish.")

    # A 'fun fact' popup or hidden tip
    if st.button("Fun Fact"):
        st.info("This project is designed to demonstrate an AI-driven approach to extending melodies.")
    st.markdown("---")

    if st.button("Proceed to Model Selection"):
        st.session_state.page = 'select_model'
        st.rerun()

# ================== Page 3: Select Model & Upload ==================
def select_model_page():
    st.title("Select Model & Upload MIDI File")

    st.markdown("""
        ### Please Upload a MIDI File
        Alternatively, pick from our sample library if you don't have one.
    """)

    uploaded_file = st.file_uploader("Upload a MIDI file (.mid):", type=["mid"])
    st.markdown("---")

    # Provide a dropdown of sample MIDIs
    samples = {
        "Familiar Tonal Snippet (input-3.mid)": "input_midis/input-3.mid",
        "Medium-Length Original Melody (input-4.mid)": "input_midis/input-4.mid",
        "Atonal, Wide-Range Melody (input-5.mid)": "input_midis/input-5.mid",
        "Long Repetitive Motif (input-6.mid)": "input_midis/input-6.mid"
    }
    chosen_sample = st.selectbox("Or select a sample:", ["None"] + list(samples.keys()))

    models = [
        ("basic", "A simple RNN-based model for melody continuation."),
        ("lookback", "Considers previous context for coherent continuations."),
        ("attention", "Uses attention mechanisms to focus on relevant patterns."),
        ("mono", "Generates monophonic melodies with basic structure.")
    ]
    model_names = [m[0] for m in models]
    chosen_model = st.selectbox("Select a model:", model_names)

    # Show model descriptions
    st.markdown("#### Model Descriptions")
    for model_key, description in models:
        st.markdown(f"- **{model_key}**: {description}")

    if st.button("Load / Use This File"):
        if uploaded_file is None and chosen_sample == "None":
            st.error("Please upload a file or select a sample first.")
            return

        # If the user has uploaded a file, prioritise that
        if uploaded_file is not None:
            st.session_state.uploaded_midi = uploaded_file.getvalue()
        else:
            # Use the sample
            sample_path = samples[chosen_sample]
            try:
                with open(sample_path, 'rb') as f:
                    st.session_state.uploaded_midi = f.read()
            except Exception as e:
                st.error(f"Could not load sample: {e}")
                return

        st.session_state.selected_model = chosen_model
        st.session_state.page = 'output'
        st.rerun()

    # If there's already an uploaded or selected sample, show it
    if st.session_state.uploaded_midi:
        st.markdown("### Current MIDI Loaded")
        # Show piano roll
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp:
                tmp.write(st.session_state.uploaded_midi)
                tmp.flush()
                df = parse_midi(tmp.name)
            fig = create_piano_roll(df)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not display piano roll: {e}")

        # Provide some form of embedded player or wave
        # We can attempt to convert to .wav and provide st.audio
        # (Though many browsers won't natively play MIDI.)
        try:
            wav_bytes = convert_midi_to_wav(st.session_state.uploaded_midi)
            st.markdown("Listen to the loaded MIDI:")
            st.audio(wav_bytes, format="audio/wav")
        except Exception as e:
            st.warning("Could not provide an audio preview.")


# ================== Page 4: Output ==================
def output_page():
    st.title("Output & Visualisations")
    st.markdown(f"Selected Model: **{st.session_state.selected_model}**")

    if not st.session_state.uploaded_midi:
        st.warning("No MIDI file found. Please go back and upload or select one.")
        return

    # Always show a piano roll for the 'generated' continuation (currently the same MIDI).
    st.markdown("### Piano Roll of Generated Continuation")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp:
            tmp.write(st.session_state.uploaded_midi)
            tmp.flush()
            df = parse_midi(tmp.name)
        fig = create_piano_roll(df)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating piano roll: {e}")

    # Provide an audio preview of the 'generated' file (converted to WAV)
    try:
        wav_bytes = convert_midi_to_wav(st.session_state.uploaded_midi)
        st.markdown("### Listen to Generated Continuation")
        st.audio(wav_bytes, format="audio/wav")
    except Exception as e:
        st.warning("Could not provide an audio preview.")

    # Download the 'generated' MIDI
    st.download_button(
        label="Download Generated MIDI",
        data=st.session_state.uploaded_midi,
        file_name="generated_continuation.mid",
        mime="audio/midi"
    )

    # Button to run first script logic (placeholder) to create a piano roll video
    if st.button("Generate Piano Roll Video"):
        try:
            # Placeholder: In a real scenario, call create_video_frames, create_video_ffmpeg etc.
            video_bytes = generate_piano_roll_video(st.session_state.uploaded_midi)
            st.download_button(
                label="Download Piano Roll Video",
                data=video_bytes,
                file_name="generated_continuation.mp4",
                mime="video/mp4"
            )
        except Exception as e:
            st.error(f"Error generating piano roll video: {e}")

    if st.button("Finish & Proceed"):
        st.session_state.page = 'closing'
        st.rerun()

# ================== Page 5: Closing ==================
def closing_page():
    # Show balloons only upon entering this page, not on interactions
    st.balloons()

    st.title("Thank You for Using the AI Music Assistant!")
    st.markdown("""
        You can run your same file with a different model if you like:
    """)

    models = [
        "basic",
        "lookback",
        "attention",
        "mono"
    ]
    new_model = st.selectbox("Choose a new model:", models, key="new_model_select")
    if st.button("Rerun with New Model"):
        st.session_state.selected_model = new_model
        st.session_state.page = 'output'
        # No balloons on this interaction
        st.rerun()

# ================== Main App Flow ==================
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
