import streamlit as st
import os
import io
import tempfile
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from mido import MidiFile
from midi2audio import FluidSynth
from tqdm import tqdm

# ================== Configuration ==================

SOUND_FONT_PATH = os.path.join(os.getcwd(), 'sounds', 'FluidR3_GM.sf2')
FPS = 30
BPM = 130
BEATS_PER_BAR = 4
SECONDS_PER_BEAT = 60 / BPM
SECONDS_PER_BAR = SECONDS_PER_BEAT * BEATS_PER_BAR

# Instantiate FluidSynth
fs = FluidSynth(sound_font=SOUND_FONT_PATH)

def parse_midi_from_bytes(midi_bytes: bytes) -> pd.DataFrame:
    """
    Parse a MIDI file (passed as bytes) and extract note information.
    Returns a pandas DataFrame with columns: note, velocity, start_time, duration.
    Copied logic from the script provided, minus path usage (we use bytes).
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp:
        tmp.write(midi_bytes)
        tmp.flush()
        tmp_path = tmp.name

    midi = MidiFile(tmp_path)
    os.remove(tmp_path)  # Clean up after reading

    notes = []
    tempo = 500000  # Default tempo (120 BPM)
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
            elif msg.type in ['note_off', 'note_on'] and msg.velocity == 0:
                for note_dict in reversed(notes):
                    if note_dict['note'] == msg.note and 'duration' not in note_dict:
                        note_end = track_times[i] * tempo / ticks_per_beat / 1_000_000
                        note_dict['duration'] = note_end - note_dict['start_time']
                        break

    # Remove notes lacking a duration
    notes = [n for n in notes if 'duration' in n]

    # Adjust so earliest note starts at 0
    if notes:
        min_start_time = min(n['start_time'] for n in notes)
        for n in notes:
            n['start_time'] -= min_start_time

    return pd.DataFrame(notes)

def generate_midi_note_names() -> dict:
    """
    Generate a mapping from MIDI note numbers to note names (A0=21 to C8=108).
    """
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']
    midi_note_names = {}
    for i in range(21, 109):
        octave = (i // 12) - 1
        note = note_names[i % 12]
        midi_note_names[i] = f"{note}{octave}"
    return midi_note_names

def create_static_piano_roll(data: pd.DataFrame) -> plt.Figure:
    """
    Create a single static piano roll figure that closely follows
    the note plotting logic from the user's script (minus the iterative frames).
    Everything is treated as if 'past' or 'active' so we see all notes at once.
    The vertical red line is omitted as requested in earlier instructions.
    """
    midi_note_names = generate_midi_note_names()

    # Map note -> note_name
    data['note_name'] = data['note'].map(midi_note_names).dropna()
    data = data[data['note_name'].notnull()].copy()

    # If no notes, return a simple figure
    if data.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No notes to display.", ha='center', va='center', fontsize=14)
        return fig

    # Determine which piano note names are actually played
    played_notes = sorted(set(data['note_name']))
    note_to_y = {note: idx for idx, note in enumerate(played_notes)}

    # Calculate the maximum time for the x-axis
    total_time_in_seconds = float(data['start_time'].max() + data['duration'].max())

    fig, ax = plt.subplots(figsize=(16, 8))

    # Basic axis setup
    ax.set_ylim(-1, len(played_notes))
    # X-axis in 'bars' from the original script's logic or just seconds?
    # The user script uses bars, but let's keep seconds for clarity:
    ax.set_xlim(0, total_time_in_seconds)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Notes")
    ax.set_title("Static Piano Roll")
    ax.set_yticks(range(len(played_notes)))
    ax.set_yticklabels(played_notes)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot all notes in a single colour
    for _, note_info in data.iterrows():
        note_number = note_info['note']
        note_name = note_info['note_name']
        if note_name not in note_to_y:
            continue
        y = note_to_y[note_name]
        start = note_info['start_time']
        duration = note_info['duration']
        ax.broken_barh([(start, duration)], (y - 0.4, 0.8),
                       facecolors='blue', edgecolors='black', linewidth=0.5)

    plt.tight_layout()
    return fig

def convert_midi_bytes_to_wav(midi_data: bytes) -> bytes:
    """
    Convert MIDI bytes to WAV bytes using FluidSynth, for audio playback.
    """
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp_midi:
        tmp_midi.write(midi_data)
        tmp_midi.flush()
        midi_path = tmp_midi.name

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wav_path = tmp_wav.name

    # Convert using FluidSynth
    fs.midi_to_audio(midi_path, wav_path)

    with open(wav_path, 'rb') as f:
        wav_bytes = f.read()

    os.remove(midi_path)
    os.remove(wav_path)
    return wav_bytes

def generate_piano_roll_video_placeholder(midi_data: bytes) -> bytes:
    """
    Placeholder for generating a piano roll video from the exact script. 
    For now, just return fake MP4 bytes.
    """
    return b"FAKE_MP4_VIDEO"

# ====== Streamlit App ======

st.set_page_config(
    page_title="AI Music Assistant",
    layout="centered",
    initial_sidebar_state="expanded"
)

if 'page' not in st.session_state:
    st.session_state.page = 'welcome'
if 'midi_source' not in st.session_state:
    st.session_state.midi_source = None  # "Upload" or "Sample"
if 'uploaded_midi' not in st.session_state:
    st.session_state.uploaded_midi = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'preview_shown' not in st.session_state:
    st.session_state.preview_shown = False

def reset_session():
    for key in ['page','midi_source','uploaded_midi','selected_model','preview_shown']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

def render_sidebar():
    st.sidebar.title("📋 Contents")
    st.sidebar.markdown("---")
    pages = ["welcome", "instructions", "select_model", "output", "closing"]
    for page in pages:
        disp = page.replace("_", " ").capitalize()
        if st.session_state.page == page:
            st.sidebar.markdown(f"### **{disp}**")
        else:
            st.sidebar.markdown(disp)
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset Session"):
        reset_session()

# ========== Pages ==========

def welcome_page():
    st.image("banner.png", use_container_width=True)
    st.title("🎵 Welcome to the AI Music Assistant 🎵")
    st.markdown("""
        This interactive tool lets you upload or select a single MIDI file, generate a 
        placeholder “continuation”, view a piano roll, and listen to or download your results.
    """)
    if st.button("Continue to Instructions"):
        st.session_state.page = 'instructions'
        st.rerun()

def instructions_page():
    st.title("Instructions & Overview")
    st.markdown("""
    1. **Select or Upload a MIDI File** (but not both at once):
       - We'll ensure you only use one approach to load MIDI.
    2. **Preview** the Piano Roll and Listen:
       - After picking a file, click "Continue to Preview". 
         That will display a static piano roll and an audio player.
    3. **Run Model & Continue**:
       - Press the button to move on to the output page, 
         where you can download the "continuation" or generate a piano roll video.
    4. **Explore**:
       - Finally, feel free to try a different model or reset the session and start over.
    """)
    st.markdown("""
    **In a future version**, different models (basic, lookback, attention, mono) will produce 
    distinct musical results. Right now, we simply return the same file for demonstration.
    """)
    if st.button("Proceed to Model Selection"):
        st.session_state.page = 'select_model'
        st.rerun()

def select_model_page():
    st.title("Select Your MIDI Source & Model")

    # Let the user choose one approach: upload or sample
    st.session_state.midi_source = st.radio(
        "How would you like to provide a MIDI file?",
        ["Upload", "Sample"],
        index=0
    )

    # If user chooses 'Upload', disable sample selection. If user chooses 'Sample', disable upload.
    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.midi_source == "Upload":
            uploaded_file = st.file_uploader("Upload a MIDI file (.mid):", type=["mid"], disabled=False)
        else:
            uploaded_file = st.file_uploader("Upload a MIDI file (.mid):", type=["mid"], disabled=True)

    with col2:
        # Provide a small dictionary of sample MIDIs
        sample_options = {
            "Familiar Tonal Snippet": "input_midis/input-3.mid",
            "Medium-Length Original Melody": "input_midis/input-4.mid",
            "Atonal, Wide-Range Melody": "input_midis/input-5.mid",
            "Long Repetitive Motif": "input_midis/input-6.mid"
        }
        if st.session_state.midi_source == "Sample":
            chosen_sample = st.selectbox("Select a Sample:", list(sample_options.keys()))
        else:
            chosen_sample = st.selectbox("Select a Sample:", list(sample_options.keys()), disabled=True)

    # Model selection
    st.markdown("### Choose a Model:")
    models = ["basic", "lookback", "attention", "mono"]
    chosen_model = st.selectbox("Model:", models)

    st.markdown("---")

    # "Continue to Preview" button logic
    if not st.session_state.preview_shown:
        if st.button("Continue to Preview"):
            # Ensure we actually have a file
            midi_bytes = None
            if st.session_state.midi_source == "Upload":
                if uploaded_file is None:
                    st.error("Please upload a MIDI file.")
                    return
                # Store
                midi_bytes = uploaded_file.read()

            else:  # Sample
                sample_path = sample_options[chosen_sample]
                try:
                    with open(sample_path, 'rb') as f:
                        midi_bytes = f.read()
                except Exception as e:
                    st.error(f"Could not load sample: {e}")
                    return

            st.session_state.uploaded_midi = midi_bytes
            st.session_state.selected_model = chosen_model
            st.session_state.preview_shown = True
            st.experimental_rerun()

    else:
        # If preview_shown is True, show the Piano Roll and Audio Player
        st.markdown(f"**Current Model**: `{st.session_state.selected_model}`")

        # Create a piano roll
        try:
            df = parse_midi_from_bytes(st.session_state.uploaded_midi)
            fig = create_static_piano_roll(df)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Failed to parse or display piano roll: {e}")

        # Create audio player
        try:
            wav_bytes = convert_midi_bytes_to_wav(st.session_state.uploaded_midi)
            st.markdown("### Audio Preview")
            st.audio(wav_bytes, format="audio/wav")
        except Exception as e:
            st.error(f"Audio preview error: {e}")

        st.markdown("---")
        if st.button("Run Model & Continue"):
            st.session_state.page = 'output'
            st.session_state.preview_shown = False
            st.rerun()

def output_page():
    st.title("Model Output")
    if not st.session_state.uploaded_midi:
        st.warning("No MIDI file is loaded. Please go back.")
        return

    st.markdown(f"**Model Selected**: `{st.session_state.selected_model}`")
    st.markdown("Below is your 'generated' continuation (actually the same file).")

    # Show piano roll again for final "continuation"
    try:
        df = parse_midi_from_bytes(st.session_state.uploaded_midi)
        fig = create_static_piano_roll(df)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Failed to create final piano roll: {e}")

    # Audio playback for the continuation
    try:
        wav_bytes = convert_midi_bytes_to_wav(st.session_state.uploaded_midi)
        st.markdown("### Listen to the Continuation:")
        st.audio(wav_bytes, format="audio/wav")
    except Exception as e:
        st.error(f"Audio playback error: {e}")

    # Download MIDI
    st.download_button(
        label="Download Generated MIDI",
        data=st.session_state.uploaded_midi,
        file_name="generated_continuation.mid",
        mime="audio/midi"
    )

    # Generate piano roll video (placeholder)
    if st.button("Generate Piano Roll Video"):
        try:
            video_bytes = generate_piano_roll_video_placeholder(st.session_state.uploaded_midi)
            st.download_button(
                label="Download Piano Roll Video",
                data=video_bytes,
                file_name="generated_continuation.mp4",
                mime="video/mp4"
            )
        except Exception as e:
            st.error(f"Video generation error: {e}")

    st.markdown("---")
    if st.button("Finish & Proceed"):
        st.session_state.page = 'closing'
        st.rerun()

def closing_page():
    st.image("closing_banner.png", use_container_width=True)
    st.title("Thank You for Using the AI Music Assistant!")
    st.markdown("""
        We hope this demonstration has given you a sense of how AI can help with musical composition. 
        Right now, all models produce the same output, but imagine each model transforming your melody 
        into something unique!
    """)

    # Show balloons only on first arrival to this page
    if 'closing_visited' not in st.session_state:
        st.session_state.closing_visited = True
        st.balloons()

    # Option to select a different model and rerun
    new_model = st.selectbox("Try a different model on the same file:", ["basic", "lookback", "attention", "mono"])
    if st.button("Rerun with New Model"):
        st.session_state.selected_model = new_model
        st.session_state.page = 'output'
        st.rerun()

# =========== Main ===========

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
