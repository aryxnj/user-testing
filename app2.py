import streamlit as st
import os
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from mido import MidiFile
from midi2audio import FluidSynth

# =================== First Script Logic (Simplified) ===================

SOUND_FONT_PATH = os.path.join(os.getcwd(), 'sounds', 'FluidR3_GM.sf2')
fs = FluidSynth(sound_font=SOUND_FONT_PATH)

def parse_midi(file_path: str) -> pd.DataFrame:
    """
    Parse a MIDI file and extract note information.
    Returns a DataFrame with columns: [note, velocity, start_time, duration].
    """
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
                notes.append({'note': msg.note, 'velocity': msg.velocity, 'start_time': note_start})
            elif msg.type in ['note_off', 'note_on'] and msg.velocity == 0:
                for note_dict in reversed(notes):
                    if note_dict['note'] == msg.note and 'duration' not in note_dict:
                        note_end = track_times[i] * tempo / ticks_per_beat / 1_000_000
                        note_dict['duration'] = note_end - note_dict['start_time']
                        break

    # Filter out notes missing duration
    notes = [n for n in notes if 'duration' in n]

    # Shift start times so the earliest note starts at 0
    if notes:
        min_start = min(n['start_time'] for n in notes)
        for n in notes:
            n['start_time'] -= min_start

    return pd.DataFrame(notes)

def generate_midi_note_names() -> dict:
    """
    Generate a mapping from MIDI note numbers to note names for A0-C8.
    """
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']
    midi_note_names = {}
    for i in range(21, 109):
        octave = (i // 12) - 1
        note = note_names[i % 12]
        midi_note_names[i] = f"{note}{octave}"
    return midi_note_names

def create_piano_roll(data: pd.DataFrame) -> plt.Figure:
    """
    Create a static piano roll plot and return the figure.
    """
    note_map = generate_midi_note_names()
    data['note_name'] = data['note'].map(note_map).dropna()

    valid_data = data[data['note_name'].notnull()]
    if valid_data.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No notes to display.", ha='center', va='center', fontsize=14)
        return fig

    # Sort notes by octave then pitch
    unique_notes = sorted(
        valid_data['note_name'].unique(),
        key=lambda x: (int(x[-1]), x[:-1])
    )
    note_to_y = {note: idx for idx, note in enumerate(unique_notes)}

    max_start = valid_data['start_time'].max()
    max_dur = valid_data['duration'].max()
    total_time = max_start + max_dur

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title("Piano Roll Visualisation")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Notes")
    ax.set_xlim(0, total_time)
    ax.set_yticks(range(len(unique_notes)))
    ax.set_yticklabels(unique_notes)
    ax.grid(True, linestyle='--', linewidth=0.5)

    for _, row in valid_data.iterrows():
        y = note_to_y[row['note_name']]
        start = row['start_time']
        duration = row['duration']
        ax.broken_barh([(start, duration)], (y - 0.4, 0.8), facecolors='blue')

    plt.tight_layout()
    return fig

def convert_midi_to_wav(midi_bytes: bytes) -> bytes:
    """
    Convert MIDI bytes to WAV bytes using FluidSynth.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp_mid:
        tmp_mid.write(midi_bytes)
        tmp_mid_path = tmp_mid.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        tmp_wav_path = tmp_wav.name

    fs.midi_to_audio(tmp_mid_path, tmp_wav_path)

    with open(tmp_wav_path, 'rb') as f:
        wav_data = f.read()

    os.remove(tmp_mid_path)
    os.remove(tmp_wav_path)
    return wav_data

def generate_piano_roll_video(midi_bytes: bytes) -> bytes:
    """
    Placeholder: In a real scenario, you'd replicate the logic for generating a video from the notes.
    """
    return b"VIDEO_PLACEHOLDER"

# ======================== Streamlit App ========================

st.set_page_config(page_title="AI Music Assistant", layout="centered")

if 'page' not in st.session_state:
    st.session_state.page = 'welcome'

if 'uploaded_midi' not in st.session_state:
    st.session_state.uploaded_midi = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'preview_shown' not in st.session_state:
    st.session_state.preview_shown = False
if 'chosen_sample' not in st.session_state:
    st.session_state.chosen_sample = "None"

def reset_session():
    for key in ['page', 'uploaded_midi', 'selected_model', 'preview_shown', 'chosen_sample', 'closing_visited']:
        if key in st.session_state:
            del st.session_state[key]
    st.experimental_rerun()

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

# ================== Welcome Page ==================
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
        st.experimental_rerun()

# ================== Instructions Page ==================
def instructions_page():
    st.title("Instructions & Overview")
    st.markdown("""
        **Workflow:**
        
        1. **Select or Upload a MIDI File**  
        2. **Choose a Model**  
        3. **Visualise & Listen** to the resulting melody  
        4. **Download** the MIDI or a video of the piano roll  
        
        ---
        **Magenta Models** used here:
        - **basic**: A simple RNN approach  
        - **lookback**: Looks back at previous measures  
        - **attention**: Focuses on relevant sections using an attention mechanism  
        - **mono**: Ensures only a single note at a time  
        
        Note: Currently, each model outputs the same unaltered file. Future improvements will offer truly unique continuations!
    """)
    st.markdown("---")
    if st.button("Proceed to Model Selection"):
        st.session_state.page = 'select_model'
        st.experimental_rerun()

# ================== Model Selection & Upload ==================
def select_model_page():
    st.title("Upload or Select a MIDI File, Then Choose a Model")

    # GREYING OUT LOGIC
    disable_sample = False
    disable_uploader = False

    # If a file is uploaded, disable the sample
    if st.session_state.uploaded_midi is not None:
        disable_sample = True

    # If a sample is chosen, disable the file uploader
    if st.session_state.chosen_sample != "None":
        disable_uploader = True

    # SAMPLES
    sample_options = {
        "Familiar Tonal Snippet": "input_midis/input-3.mid",
        "Medium-Length Original Melody": "input_midis/input-4.mid",
        "Atonal, Wide-Range Melody": "input_midis/input-5.mid",
        "Long Repetitive Motif": "input_midis/input-6.mid"
    }
    sample_list = ["None"] + list(sample_options.keys())

    # FILE UPLOADER
    uploaded_file = st.file_uploader(
        "Upload a MIDI file (.mid):",
        type=["mid"],
        disabled=disable_uploader
    )

    # SAMPLE SELECT BOX
    chosen_sample = st.selectbox(
        "Or select a sample MIDI:",
        sample_list,
        index=sample_list.index(st.session_state.chosen_sample)
        if st.session_state.chosen_sample in sample_list
        else 0,
        disabled=disable_sample
    )

    # Immediately respond if user changes the sample
    if chosen_sample != st.session_state.chosen_sample:
        st.session_state.chosen_sample = chosen_sample
        st.session_state.preview_shown = False  # Reset preview
        if chosen_sample != "None":
            # Clear any uploaded file
            st.session_state.uploaded_midi = None
        st.experimental_rerun()

    # Immediately respond if user uploads a file
    if uploaded_file is not None:
        st.session_state.uploaded_midi = uploaded_file.getvalue()
        st.session_state.preview_shown = False  # Reset preview
        if st.session_state.chosen_sample != "None":
            st.session_state.chosen_sample = "None"
        st.experimental_rerun()

    # MODEL SELECT
    model_list = ["basic", "lookback", "attention", "mono"]
    chosen_model = st.selectbox("Select a model:", model_list)
    st.session_state.selected_model = chosen_model

    st.markdown("---")
    # If preview_shown is False, offer a "Continue to preview" button
    if not st.session_state.preview_shown:
        if st.button("Continue to preview"):
            # Check if user has a file or sample
            if (st.session_state.uploaded_midi is None) and (st.session_state.chosen_sample == "None"):
                st.error("Please either upload a MIDI file or select a sample first.")
            else:
                st.session_state.preview_shown = True
                st.experimental_rerun()
    else:
        # PREVIEW (piano roll + audio)
        st.markdown("### Preview of Selected MIDI")
        if st.session_state.uploaded_midi:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp:
                    tmp.write(st.session_state.uploaded_midi)
                    tmp.flush()
                    df = parse_midi(tmp.name)

                if df.empty:
                    st.warning("No notes found in this MIDI file.")
                else:
                    fig = create_piano_roll(df)
                    st.pyplot(fig)

                # Attempt to create WAV for audio preview
                try:
                    wav_bytes = convert_midi_to_wav(st.session_state.uploaded_midi)
                    st.markdown("#### Listen to Your MIDI:")
                    st.audio(wav_bytes, format="audio/wav")
                except Exception as e:
                    st.warning(f"Could not create audio preview: {e}")
            except Exception as e:
                st.error(f"Error parsing MIDI: {e}")

        st.markdown("---")
        # Button to proceed
        if st.button("Run Model & Continue"):
            st.session_state.page = 'output'
            st.experimental_rerun()

# ================== Output Page ==================
def output_page():
    st.title("Your Model Output")
    if not st.session_state.uploaded_midi:
        st.warning("No MIDI file found. Please go back and select or upload one.")
        return

    st.markdown(f"**Selected Model**: `{st.session_state.selected_model}`")
    st.markdown("""
        Below is the 'continuation' for your MIDI file. Currently, our system just returns the same file 
        rather than generating a truly new continuation. Future improvements will allow each model 
        to produce a unique transformation or extension.
    """)

    # Generate piano roll
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
        fig = create_piano_roll(df)
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
        st.experimental_rerun()

# ================== Closing Page ==================
def closing_page():
    closing_banner_path = "closing_banner.png"
    if os.path.exists(closing_banner_path):
        st.image(closing_banner_path, use_container_width=True)

    st.title("Thank You for Using the AI Music Assistant!")
    st.markdown("""
        We hope you enjoyed exploring our simple AI-based MIDI tool. 
        If you wish, you can select a different model below and run the same file again 
        without any additional fireworks.
    """)

    # Show balloons only once
    if 'closing_visited' not in st.session_state:
        st.session_state.closing_visited = True
        st.balloons()

    new_model = st.selectbox("Select a different model to rerun:", ["basic", "lookback", "attention", "mono"])
    if st.button("Rerun with New Model"):
        st.session_state.selected_model = new_model
        st.session_state.page = 'output'
        st.experimental_rerun()

# ================== Main Flow ==================
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
