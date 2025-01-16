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

    # Sort the unique notes actually played
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
    Convert MIDI bytes to WAV bytes using FluidSynth.
    """
    import tempfile
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
# We'll use a small state var for previewing
if 'show_preview' not in st.session_state:
    st.session_state.show_preview = False

def reset_session():
    keys = ['page', 'uploaded_midi', 'selected_model', 'show_preview']
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]
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
        This interactive app allows you to upload or select a MIDI file and explore 
        how a simple AI model can generate a melodic continuation. Experiment with 
        different models, view an on-screen piano roll, and download your results.
    """)

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
        Here’s how you can use this AI Music Assistant:

        1. **Go to 'Select Model'** and pick either a MIDI file from our library **or** upload your own (we'll disable one option if you choose the other).
        2. **Once loaded**, you'll see a preview (piano roll + audio). 
        3. **Run the model** to create a 'continuation' (currently just returns your file).
        4. **Download** the final MIDI or a placeholder video if you want.

        Then, you can head to the closing page where you can pick a different model to try again.
    """)
    st.markdown("---")
    if st.button("Proceed to Model Selection"):
        st.session_state.page = 'select_model'
        st.rerun()

# ================== Page 3: Select Model & Upload ==================
def select_model_page():
    st.title("Select a Model & MIDI File")

    st.markdown("""
        **Please choose either to upload your own MIDI file or select one of our samples.** 
        The other option will be disabled automatically once you interact with one.
    """)

    # We'll store which option the user chose:
    if 'source_option' not in st.session_state:
        st.session_state.source_option = None

    col_left, col_right = st.columns(2)

    # Left column for user upload
    with col_left:
        st.subheader("1) Upload Your MIDI")
        # If user has chosen sample, disable this upload
        disabled_upload = (st.session_state.source_option == "sample")
        uploaded_file = st.file_uploader(
            "Upload a MIDI file (.mid):", 
            type=["mid"],
            disabled=disabled_upload
        )
        if uploaded_file and st.session_state.source_option != "sample":
            # The user is going with upload
            st.session_state.source_option = "upload"

    # Right column for sample selection
    with col_right:
        st.subheader("2) Or Select a Sample")
        sample_options = [
            "None",  # default
            "Familiar Tonal Snippet",
            "Medium-Length Original Melody",
            "Atonal, Wide-Range Melody",
            "Long Repetitive Motif"
        ]
        disabled_sample = (st.session_state.source_option == "upload")
        chosen_sample = st.selectbox(
            "Pick a sample from our library:", 
            sample_options, 
            disabled=disabled_sample
        )
        if chosen_sample != "None" and st.session_state.source_option != "upload":
            st.session_state.source_option = "sample"

    st.markdown("---")

    st.subheader("Choose a Model")
    model_list = ["basic", "lookback", "attention", "mono"]
    chosen_model = st.selectbox("Model:", model_list)

    # If the user hasn't interacted with either upload or sample, we do nothing
    if st.button("Continue to Preview"):
        if st.session_state.source_option is None:
            st.error("Please either upload a MIDI file or select a sample.")
            return

        # If user upload is chosen
        if st.session_state.source_option == "upload":
            if not uploaded_file:
                st.error("Please upload a file.")
                return
            st.session_state.uploaded_midi = uploaded_file.getvalue()

        # If sample is chosen
        elif st.session_state.source_option == "sample":
            if chosen_sample == "None":
                st.error("Please select a valid sample.")
                return
            # Map them to actual paths
            sample_map = {
                "Familiar Tonal Snippet": "input_midis/input-3.mid",
                "Medium-Length Original Melody": "input_midis/input-4.mid",
                "Atonal, Wide-Range Melody": "input_midis/input-5.mid",
                "Long Repetitive Motif": "input_midis/input-6.mid"
            }
            file_path = sample_map[chosen_sample]
            try:
                with open(file_path, 'rb') as f:
                    st.session_state.uploaded_midi = f.read()
            except Exception as e:
                st.error(f"Failed to load sample: {e}")
                return

        st.session_state.selected_model = chosen_model
        # Show the preview
        st.session_state.show_preview = True
        st.experimental_rerun()

    # Preview section (once user has pressed the button)
    if st.session_state.show_preview and st.session_state.uploaded_midi:
        st.markdown("---")
        st.markdown("### Preview of Your MIDI File")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp:
                tmp.write(st.session_state.uploaded_midi)
                tmp.flush()
                df = parse_midi(tmp.name)
            if df.empty:
                st.warning("No notes found in this MIDI or unable to parse.")
            else:
                fig = create_piano_roll(df)
                st.pyplot(fig)

            # Attempt audio preview
            try:
                wav_data = convert_midi_to_wav(st.session_state.uploaded_midi)
                st.markdown("#### Listen to Your MIDI:")
                st.audio(wav_data, format="audio/wav")
            except Exception as e:
                st.warning(f"Could not generate audio preview: {e}")
        except Exception as e:
            st.error(f"Error loading MIDI: {e}")

        # Button to run model
        if st.button("Run Model & Continue"):
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
    if 'closing_visited' not in st.session_state:
        st.session_state.closing_visited = True
        st.balloons()

    st.image("closing_banner.png", use_container_width=True)
    st.title("Thank You for Using the AI Music Assistant!")
    st.markdown("""
        We hope you enjoyed exploring our simple AI-based MIDI tool. 
        If you wish, you can select a different model below and run the same file again.
    """)

    models = ["basic", "lookback", "attention", "mono"]
    new_model = st.selectbox("Select a different model to rerun:", models)
    if st.button("Rerun with New Model"):
        st.session_state.selected_model = new_model
        st.session_state.page = 'output'
        # No new balloons
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
