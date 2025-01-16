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

# ============================================
# ====== Copied Logic from First Script ======
# ============================================

SOUND_FONT_PATH = os.path.join(os.getcwd(), 'sounds', 'FluidR3_GM.sf2')
FPS = 30
BPM = 130
BEATS_PER_BAR = 4
SECONDS_PER_BEAT = 60 / BPM
SECONDS_PER_BAR = SECONDS_PER_BEAT * BEATS_PER_BAR

# Instantiate FluidSynth for potential audio rendering
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
                for note_dict in reversed(notes):
                    if note_dict['note'] == msg.note and 'duration' not in note_dict:
                        note_end = track_times[i] * tempo / ticks_per_beat / 1_000_000
                        note_dict['duration'] = note_end - note_dict['start_time']
                        break

    # Remove any note without a duration
    notes = [n for n in notes if 'duration' in n]

    # Align start times so earliest note is at 0
    if notes:
        min_start_time = min(n['start_time'] for n in notes)
        for n in notes:
            n['start_time'] -= min_start_time

    return pd.DataFrame(notes)

def generate_midi_note_names() -> dict:
    """
    Generate a mapping from MIDI note numbers to note names.
    """
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']
    midi_note_names = {}
    for i in range(21, 109):  # Piano range A0 (21) to C8 (108)
        octave = (i // 12) - 1
        note = note_names[i % 12]
        midi_note_names[i] = f"{note}{octave}"
    return midi_note_names

def create_piano_roll(data: pd.DataFrame) -> plt.Figure:
    """
    Create a static piano roll plot (no red line) and return the matplotlib Figure.
    This logic matches the user's original first script approach (minus the red line).
    """
    midi_note_names = generate_midi_note_names()

    # Map 'note' -> 'note_name'
    data['note_name'] = data['note'].map(midi_note_names).dropna()
    valid_data = data[data['note_name'].notnull()].copy()
    if valid_data.empty:
        # Return an empty figure if no notes
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No notes to display.", ha='center', va='center', fontsize=14)
        return fig

    # Sort the unique notes actually played, from low to high
    unique_notes_used = sorted(
        valid_data['note_name'].unique(),
        key=lambda x: (int(x[-1]), x[:-1])
    )

    # Create y-axis mapping
    note_to_y = {note: idx for idx, note in enumerate(unique_notes_used)}

    max_start_time = valid_data['start_time'].max()
    max_duration = valid_data['duration'].max()
    total_time_in_seconds = max_start_time + max_duration

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_yticks(range(len(unique_notes_used)))
    ax.set_yticklabels(unique_notes_used)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Notes")
    ax.set_title("Piano Roll Visualisation")
    ax.set_xlim(0, total_time_in_seconds)
    ax.grid(True, linestyle='--', linewidth=0.5)

    for idx, row in valid_data.iterrows():
        note_name = row['note_name']
        y = note_to_y[note_name]
        start = row['start_time']
        duration = row['duration']
        ax.broken_barh([(start, duration)], (y - 0.4, 0.8), facecolors='blue')

    plt.tight_layout()
    return fig

def convert_midi_to_wav(midi_data: bytes) -> bytes:
    """
    Convert MIDI bytes to WAV bytes using FluidSynth.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp_mid:
        tmp_mid.write(midi_data)
        tmp_mid_path = tmp_mid.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        tmp_wav_path = tmp_wav.name

    # Convert using FluidSynth
    fs.midi_to_audio(tmp_mid_path, tmp_wav_path)

    with open(tmp_wav_path, 'rb') as f:
        wav_bytes = f.read()

    # Cleanup
    os.remove(tmp_mid_path)
    os.remove(tmp_wav_path)
    return wav_bytes

def generate_piano_roll_video(midi_bytes: bytes) -> bytes:
    """
    Placeholder function to simulate the creation of a piano roll video (MP4).
    In a real scenario, we'd replicate the create_video_frames and create_video_ffmpeg logic.
    """
    return b"VIDEO_PLACEHOLDER"


# =======================================
# ========== Streamlit App =============
# =======================================

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
        This interactive app allows you to upload (or select) a MIDI file and explore 
        how a simple AI model can generate a melodic continuation. Experiment with 
        different models, view an on-screen piano roll, and download your results. 
        We hope you enjoy this small taste of AI-assisted composition!
    """)

    # Put example videos in a dropdown
    with st.expander("View Example Videos of Input/Output"):
        st.markdown("**Input: Familiar Tonal Snippet (input-3)**")
        st.video("output_videos/input-3.mp4")

        st.markdown("**Output: Continuation by Mono Model (output-3-mono)**")
        st.video("output_videos/output-3-mono.mp4")

        st.markdown("**Output: Continuation by Lookback Model (output-3-lookback)**")
        st.video("output_videos/output-3-lookback.mp4")

    st.markdown("---")
    if st.button("Continue to Instructions"):
        st.session_state.page = 'instructions'
        st.rerun()

# ================== Page 2: Instructions ==================
def instructions_page():
    st.title("Instructions & Overview")
    st.markdown("""
        This AI Music Assistant offers a straightforward workflow:

        1. **Select or Upload a MIDI File**: 
           - You can pick from our provided sample MIDI files (like the familiar snippet or atonal example), 
             or upload your own `.mid` file to explore.
        2. **Choose a Model**: 
           - We'll run your chosen file through an AI model (placeholder logic for now). 
           - In the future, each model will generate unique transformations and continuations.
        3. **Visualise & Listen**: 
           - Instantly view the piano roll representation of your melody or the generated continuation. 
           - Listen to it through an audio player, so there's no need to download if you're just previewing.
        4. **Download**: 
           - If you like, you can download both the MIDI version of the result and a generated video 
             showing the piano roll in motion (placeholder for now).

        Below, you'll also find a quick overview of the Magenta models we reference:
    """)

    with st.expander("A Quick Tour of Magenta Models"):
        st.markdown("""
        - **Basic RNN**: A simple recurrent network that generates basic melodic sequences.
        - **Lookback RNN**: Incorporates 'lookback' steps to track patterns from earlier measures.
        - **Attention**: Leverages attention mechanisms to 'focus' on relevant parts of the melody context.
        - **Mono**: A monophonic model that ensures only one note sounds at a time.

        In practice, these models can produce a variety of results, from simple to elaborate. 
        Here, however, we’re using placeholders—so the actual continuations won't differ yet!
        """)

    st.markdown("""
        **Next Steps**:
        - Click below to proceed and begin selecting your MIDI file.
    """)

    # Some interactive text box or pop-up
    st.markdown("If you have any questions along the way, feel free to explore or hover over hints.")

    st.markdown("---")
    if st.button("Proceed to Model Selection"):
        st.session_state.page = 'select_model'
        st.rerun()

# ================== Page 3: Select Model & Upload ==================
def select_model_page():
    st.title("Upload or Select a MIDI File, Then Choose a Model")

    st.markdown("""
        **Step 1**: Upload your MIDI file using the uploader below **or** select one of our samples.
    """)

    # File upload
    uploaded_file = st.file_uploader("Upload a MIDI file (.mid):", type=["mid"])

    # Sample MIDIs (short names)
    sample_options = {
        "Familiar Tonal Snippet": "input_midis/input-3.mid",
        "Medium-Length Original Melody": "input_midis/input-4.mid",
        "Atonal, Wide-Range Melody": "input_midis/input-5.mid",
        "Long Repetitive Motif": "input_midis/input-6.mid"
    }

    st.markdown("**Or Select a Sample MIDI:**")
    chosen_sample = st.selectbox("Choose from our library:", ["None"] + list(sample_options.keys()))

    # Model selection (no descriptions on this page)
    st.markdown("**Step 2**: Choose a Model.")
    model_list = ["basic", "lookback", "attention", "mono"]
    chosen_model = st.selectbox("Select a model:", model_list)

    st.markdown("---")

    if st.button("Confirm & Preview"):
        # Check if a user file or sample was chosen
        if not uploaded_file and chosen_sample == "None":
            st.error("Please either upload a MIDI file or choose a sample!")
            return

        # Priority to user's uploaded file
        if uploaded_file is not None:
            st.session_state.uploaded_midi = uploaded_file.getvalue()
        else:
            # Use the sample
            sample_path = sample_options[chosen_sample]
            try:
                with open(sample_path, 'rb') as f:
                    st.session_state.uploaded_midi = f.read()
            except Exception as e:
                st.error(f"Unable to load sample: {e}")
                return

        st.session_state.selected_model = chosen_model
        st.success(f"MIDI file loaded successfully! (Model: {chosen_model})")

        # Display the newly selected/loaded MIDI file's piano roll + audio preview
        if st.session_state.uploaded_midi:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp:
                tmp.write(st.session_state.uploaded_midi)
                tmp.flush()
                df = parse_midi(tmp.name)

            if df.empty:
                st.warning("No notes in this MIDI file or unable to parse.")
            else:
                fig = create_piano_roll(df)
                st.pyplot(fig)
                try:
                    wav_bytes = convert_midi_to_wav(st.session_state.uploaded_midi)
                    st.markdown("### Listen to Your Selection:")
                    st.audio(wav_bytes, format="audio/wav")
                except Exception as e:
                    st.warning(f"Could not create audio preview: {e}")

    st.markdown("---")
    if st.button("Run Model & Continue"):
        if st.session_state.uploaded_midi is None:
            st.error("No MIDI selected or uploaded yet.")
            return
        st.session_state.page = 'output'
        st.rerun()

# ================== Page 4: Output ==================
def output_page():
    st.title("Your Model Output")
    if not st.session_state.uploaded_midi:
        st.warning("No MIDI file found. Please go back and select/upload one.")
        return

    st.markdown(f"**Selected Model**: `{st.session_state.selected_model}`")
    st.markdown("""
        Below is the 'continuation' for your MIDI file. Currently, our system just returns the same file 
        rather than generating a truly new continuation. Future improvements will allow each model 
        to produce a unique transformation or extension.
    """)

    # Show the 'generated' continuation's piano roll
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp:
        tmp.write(st.session_state.uploaded_midi)
        tmp.flush()
        df = parse_midi(tmp.name)

    if df.empty:
        st.warning("No valid notes found in the final output.")
    else:
        fig = create_piano_roll(df)
        st.pyplot(fig)

    # Audio preview
    try:
        wav_bytes = convert_midi_to_wav(st.session_state.uploaded_midi)
        st.markdown("### Listen to the Continuation:")
        st.audio(wav_bytes, format="audio/wav")
    except Exception as e:
        st.warning(f"Unable to generate audio preview: {e}")

    # MIDI download
    st.download_button(
        label="Download Generated MIDI",
        data=st.session_state.uploaded_midi,
        file_name="generated_continuation.mid",
        mime="audio/midi"
    )

    # Piano roll video generation placeholder
    if st.button("Generate Piano Roll Video (.mp4)"):
        try:
            video_bytes = generate_piano_roll_video(st.session_state.uploaded_midi)
            st.download_button(
                label="Download Piano Roll Video",
                data=video_bytes,
                file_name="generated_continuation.mp4",
                mime="video/mp4"
            )
        except Exception as e:
            st.error(f"Error generating piano roll video: {e}")

    st.markdown("---")
    if st.button("Finish & Proceed"):
        st.session_state.page = 'closing'
        st.rerun()

# ================== Page 5: Closing ==================
def closing_page():
    # Show a closing banner
    st.image("closing_banner.png", use_container_width=True)
    st.title("Thank You for Using the AI Music Assistant!")
    st.markdown("""
        We hope you enjoyed exploring our simple AI-based MIDI tool. 
        If you wish, you can select a different model below and run the same file again 
        (without the celebratory balloons). 
    """)

    # Show balloons only upon first arrival on this page
    if 'closing_visited' not in st.session_state:
        st.session_state.closing_visited = True
        st.balloons()

    new_model = st.selectbox("Select a different model to rerun:", ["basic", "lookback", "attention", "mono"])
    if st.button("Rerun with New Model"):
        st.session_state.selected_model = new_model
        st.session_state.page = 'output'
        st.rerun()

# ============== Main Flow ==============
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
