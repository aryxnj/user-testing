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

def create_static_piano_roll(data, total_time_in_seconds):
    """
    Create a static piano roll plot with all notes in a faded blue colour.
    Returns a matplotlib Figure.
    """
    midi_note_names = generate_midi_note_names()
    
    # Define all piano notes from A0 to C8
    unique_notes = [midi_note_names[i] for i in range(21, 109)]
    
    # Create a mapping for y-axis positions
    note_to_y = {note: idx for idx, note in enumerate(unique_notes)}
    
    # Map data['note'] to note names, filter out any notes outside the piano range
    data['note_name'] = data['note'].map(midi_note_names).dropna()
    
    # Determine the octaves played based on the notes in the MIDI file
    played_notes = data['note'].unique()
    played_octaves = sorted(set((note // 12) - 1 for note in played_notes if 21 <= note <= 108))
    
    # Filter unique_notes to include only the octaves played
    filtered_notes = []
    for octave in played_octaves:
        for note in ['C', 'C#', 'D', 'D#', 'E', 'F',
                    'F#', 'G', 'G#', 'A', 'A#', 'B']:
            try:
                note_index = ['C', 'C#', 'D', 'D#', 'E', 'F',
                              'F#', 'G', 'G#', 'A', 'A#', 'B'].index(note)
            except ValueError:
                continue  # Skip invalid notes
            midi_num = (octave + 1) * 12 + note_index
            if 21 <= midi_num <= 108:
                filtered_notes.append(f"{note}{octave}")
    
    # Update y-axis mappings based on filtered_notes
    unique_notes = filtered_notes
    note_to_y = {note: idx for idx, note in enumerate(unique_notes)}

    # Create a figure for the static piano roll
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_ylim(-1, len(unique_notes))
    ax.set_xlim(0, total_time_in_seconds / SECONDS_PER_BAR)  # X-axis in bars
    ax.set_xlabel("Bars")
    ax.set_ylabel("Notes")
    ax.set_title("Static Piano Roll Visualisation")
    ax.set_yticks(range(len(unique_notes)))
    ax.set_yticklabels(unique_notes)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot all notes as faded blue boxes
    for _, note_info in data.iterrows():
        note_number = note_info['note']
        note_name = midi_note_names.get(note_number, None)
        if note_name is None or note_name not in unique_notes:
            continue  # Skip notes outside the filtered range
        y = note_to_y[note_name]
        # Calculate start_bar and duration_bars
        start_bar = note_info['start_time'] / SECONDS_PER_BAR
        duration_bars = note_info['duration'] / SECONDS_PER_BAR
        ax.broken_barh([(start_bar, duration_bars)],
                      (y - 0.4, 0.8),
                      facecolors=(0, 0, 1, 0.3),  # Faded blue with alpha=0.3
                      edgecolors='black',
                      linewidth=0.5)
    
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

    fs.midi_to_audio(tmp_mid_path, tmp_wav_path)

    with open(tmp_wav_path, 'rb') as f:
        wav_bytes = f.read()

    os.remove(tmp_mid_path)
    os.remove(tmp_wav_path)
    return wav_bytes

def generate_piano_roll_video(midi_bytes: bytes) -> bytes:
    """
    Placeholder for creating a piano roll video from the given MIDI bytes.
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

# We'll store the raw MIDI bytes and model choice in session state
if 'uploaded_midi' not in st.session_state:
    st.session_state.uploaded_midi = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None

def reset_session():
    for key in ['page', 'uploaded_midi', 'selected_model', 'preview_shown', 'chosen_sample']:
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

        1. **Select or Upload a MIDI File**  
           You can pick from our provided sample MIDI files (like a familiar snippet, or an atonal example), 
           or upload your own `.mid` file to explore.

        2. **Choose a Model**  
           We'll run your chosen file through a placeholder AI model. In the future, each model 
           will generate unique transformations and continuations.

        3. **Visualise & Listen**  
           Instantly view the piano roll representation of your melody or the generated continuation, 
           and listen to it via an audio player—no extra downloads required for a quick preview.

        4. **Download**  
           If you like, you can download both the MIDI version of the result and a generated video 
           showing the piano roll in motion (currently a placeholder).

        ---
        **Magenta Model Notes**:  
        - **basic**: A simple RNN-based approach.  
        - **lookback**: Considers prior measures for context.  
        - **attention**: Uses attention mechanisms to focus on relevant sections.  
        - **mono**: A monophonic model ensuring only one note sounds at a time.

        ---
        Click below when you're ready to choose a MIDI and model.
    """)

    st.markdown("---")
    if st.button("Proceed to Model Selection"):
        st.session_state.page = 'select_model'
        st.rerun()

# ================== Page 3: Select Model & Upload ==================
def select_model_page():
    st.title("Upload or Select a MIDI File, Then Choose a Model")

    # Toggle between uploading a file or using a sample
    input_method = st.radio("Select input method:", ["Upload MIDI", "Use Sample"])

    # Depending on the choice, show the corresponding input and grey out the other
    if input_method == "Upload MIDI":
        uploaded_file = st.file_uploader("Upload a MIDI file (.mid):", type=["mid"])
        # Hide sample selection when uploading
        sample_options = {}
        chosen_sample = None
    else:
        uploaded_file = None
        sample_options = {
            "Familiar Tonal Snippet": "input_midis/input-3.mid",
            "Medium-Length Original Melody": "input_midis/input-4.mid",
            "Atonal, Wide-Range Melody": "input_midis/input-5.mid",
            "Long Repetitive Motif": "input_midis/input-6.mid"
        }
        chosen_sample = st.selectbox("Choose a sample MIDI:", list(sample_options.keys()))

    st.markdown("**Select a Model:**")
    model_list = ["basic", "lookback", "attention", "mono"]
    chosen_model = st.selectbox("Select a model:", model_list)

    if st.button("Continue to Preview"):
        # Load MIDI data based on input method
        if input_method == "Upload MIDI":
            if uploaded_file is None:
                st.error("Please upload a MIDI file.")
                return
            st.session_state.uploaded_midi = uploaded_file.getvalue()
        else:
            sample_path = sample_options[chosen_sample]
            try:
                with open(sample_path, 'rb') as f:
                    st.session_state.uploaded_midi = f.read()
            except Exception as e:
                st.error(f"Unable to load sample: {e}")
                return

        st.session_state.selected_model = chosen_model

        # Display piano roll and audio preview for the loaded MIDI
        if st.session_state.uploaded_midi:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp:
                tmp.write(st.session_state.uploaded_midi)
                tmp.flush()
                df = parse_midi(tmp.name)
            if not df.empty:
                total_time_in_seconds = df['start_time'].max() + df['duration'].max()
                fig = create_static_piano_roll(df, total_time_in_seconds)
                st.pyplot(fig)
                try:
                    wav_bytes = convert_midi_to_wav(st.session_state.uploaded_midi)
                    st.markdown("### Listen to Your Selection:")
                    st.audio(wav_bytes, format="audio/wav")
                except Exception as e:
                    st.warning(f"Could not create audio preview: {e}")
            else:
                st.warning("No valid notes found in the selected MIDI.")

        # Show "Run Model & Continue" button after previewing
        st.button("Run Model & Continue", on_click=lambda: (setattr(st.session_state, 'page', 'output'), st.rerun()))


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
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp:
            tmp.write(st.session_state.uploaded_midi)
            tmp.flush()
            df = parse_midi(tmp.name)
    except Exception as e:
        df = pd.DataFrame()
        st.error(f"Error loading MIDI for output: {e}")

    if df.empty:
        st.warning("No valid notes found in this MIDI.")
    else:
        total_time_in_seconds = df['start_time'].max() + df['duration'].max()
        fig = create_static_piano_roll(df, total_time_in_seconds)
        st.pyplot(fig)

    # Audio preview
    try:
        wav_bytes = convert_midi_to_wav(st.session_state.uploaded_midi)
        st.markdown("### Listen to the Continuation:")
        st.audio(wav_bytes, format="audio/wav")
    except Exception as e:
        st.warning(f"Could not provide an audio preview: {e}")

    # MIDI download
    st.download_button(
        label="Download Generated MIDI",
        data=st.session_state.uploaded_midi,
        file_name="generated_continuation.mid",
        mime="audio/midi"
    )

    # Piano roll video placeholder
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
    if os.path.exists("closing_banner.png"):
        st.image("closing_banner.png", use_container_width=True)

    st.title("Thank You for Using the AI Music Assistant!")
    st.markdown("""
        We hope you enjoyed exploring our simple AI-based MIDI tool. 
        If you wish, you can select a different model below and run the same file again 
        without any additional fireworks.
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
