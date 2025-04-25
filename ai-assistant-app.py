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
from models.generate_lstm_continuation import run_lstm_continuation
from models.generate_markov_continuation import run_markov_continuation

# ===============================
# Piano Roll Videos Script Logic
# ===============================

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
    
    data['note_name'] = data['note'].map(midi_note_names).dropna()
    played_notes = data['note'].unique()
    played_octaves = sorted(set((note // 12) - 1 for note in played_notes if 21 <= note <= 108))

    filtered_notes = []
    for octave in played_octaves:
        for note in ['C', 'C#', 'D', 'D#', 'E', 'F',
                     'F#', 'G', 'G#', 'A', 'A#', 'B']:
            try:
                note_index = ['C', 'C#', 'D', 'D#', 'E',
                              'F', 'F#', 'G', 'G#', 'A',
                              'A#', 'B'].index(note)
            except ValueError:
                continue
            midi_num = (octave + 1) * 12 + note_index
            if 21 <= midi_num <= 108:
                filtered_notes.append(f"{note}{octave}")

    unique_notes = filtered_notes
    note_to_y = {note: idx for idx, note in enumerate(unique_notes)}

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_ylim(-1, len(unique_notes))
    ax.set_xlim(0, total_time_in_seconds / SECONDS_PER_BAR)
    ax.set_xlabel("Bars")
    ax.set_ylabel("Notes")
    ax.set_title("Static Piano Roll Visualisation")
    ax.set_yticks(range(len(unique_notes)))
    ax.set_yticklabels(unique_notes)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    for _, note_info in data.iterrows():
        note_number = note_info['note']
        note_name = midi_note_names.get(note_number, None)
        if note_name is None or note_name not in unique_notes:
            continue
        y = note_to_y[note_name]
        start_bar = note_info['start_time'] / SECONDS_PER_BAR
        duration_bars = note_info['duration'] / SECONDS_PER_BAR
        ax.broken_barh([(start_bar, duration_bars)],
                       (y - 0.4, 0.8),
                       facecolors=(0, 0, 1, 0.3),
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
    Create a piano roll video including a moving red line and bold 'active' notes,
    replicating the style of the original script. Past notes are faded, upcoming notes
    are faded, and current notes are solid. The static piano roll remains unchanged.
    """
    import tempfile
    import shutil

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        st.error("FFmpeg not found. Please install FFmpeg or ensure it is in your system PATH.")
        return b""

    # Write MIDI bytes to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp_mid:
        tmp_mid.write(midi_bytes)
        tmp_mid_path = tmp_mid.name

    midi_data = parse_midi(tmp_mid_path)
    if midi_data.empty:
        return b""

    total_time_in_seconds = midi_data['start_time'].max() + midi_data['duration'].max()

    # Convert MIDI to WAV for final audio track
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        tmp_wav_path = tmp_wav.name
    fs.midi_to_audio(tmp_mid_path, tmp_wav_path)

    # Prepare frame generation
    frames_dir = tempfile.mkdtemp()
    frame_count = int(total_time_in_seconds * FPS)
    progress_bar = st.progress(0)

    # Sort notes by start time, track active notes as time advances
    data_sorted = midi_data.sort_values('start_time').reset_index(drop=True)
    current_index = 0
    active_notes = []
    past_notes = []

    midi_note_names = generate_midi_note_names()

    # Define all piano notes from A0 to C8
    unique_notes = [midi_note_names[i] for i in range(21, 109)]
    note_to_y = {note: idx for idx, note in enumerate(unique_notes)}

    # Filter out-of-range notes
    data_sorted['note_name'] = data_sorted['note'].map(midi_note_names).dropna()

    # Determine the octaves played
    played_notes = data_sorted['note'].unique()
    played_octaves = sorted(set((n // 12) - 1 for n in played_notes if 21 <= n <= 108))

    filtered_notes = []
    for octave in played_octaves:
        for note in ['C', 'C#', 'D', 'D#', 'E', 'F',
                     'F#', 'G', 'G#', 'A', 'A#', 'B']:
            try:
                note_index = ['C', 'C#', 'D', 'D#', 'E',
                              'F', 'F#', 'G', 'G#', 'A',
                              'A#', 'B'].index(note)
            except ValueError:
                continue
            midi_num = (octave + 1) * 12 + note_index
            if 21 <= midi_num <= 108:
                filtered_notes.append(f"{note}{octave}")

    unique_notes = filtered_notes
    note_to_y = {note: idx for idx, note in enumerate(unique_notes)}

    for frame_idx in range(frame_count):
        current_time = frame_idx / FPS
        current_bar = current_time / SECONDS_PER_BAR

        # Add newly reached notes to active list
        while current_index < len(data_sorted) and data_sorted.loc[current_index, 'start_time'] <= current_time:
            note = data_sorted.loc[current_index, 'note']
            velocity = data_sorted.loc[current_index, 'velocity']
            duration = data_sorted.loc[current_index, 'duration']
            end_time = data_sorted.loc[current_index, 'start_time'] + duration
            active_notes.append({
                'note': note,
                'velocity': velocity,
                'start_time': data_sorted.loc[current_index, 'start_time'],
                'duration': duration,
                'end_time': end_time
            })
            current_index += 1

        # Move notes that have ended into past_notes
        for note_info in active_notes[:]:
            if note_info['end_time'] <= current_time:
                past_notes.append(note_info)
                active_notes.remove(note_info)

        # Identify upcoming notes (those that haven’t started yet)
        upcoming_notes = data_sorted[data_sorted['start_time'] > current_time]

        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_ylim(-1, len(unique_notes))
        ax.set_xlim(0, total_time_in_seconds / SECONDS_PER_BAR)
        ax.set_xlabel("Bars")
        ax.set_ylabel("Notes")
        ax.set_title("Generated Piano Roll with Red Time Line")
        ax.set_yticks(range(len(unique_notes)))
        ax.set_yticklabels(unique_notes)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Past notes = faded
        for note_info in past_notes:
            note_number = note_info['note']
            note_name = midi_note_names.get(note_number, None)
            if note_name is None or note_name not in unique_notes:
                continue
            y = note_to_y[note_name]
            start_bar = note_info['start_time'] / SECONDS_PER_BAR
            duration_bars = note_info['duration'] / SECONDS_PER_BAR
            ax.broken_barh([(start_bar, duration_bars)],
                           (y - 0.4, 0.8),
                           facecolors=(0, 0, 1, 0.3),
                           edgecolors='black',
                           linewidth=0.5)

        # Active notes = bold, fully opaque
        for note_info in active_notes:
            note_number = note_info['note']
            note_name = midi_note_names.get(note_number, None)
            if note_name is None or note_name not in unique_notes:
                continue
            y = note_to_y[note_name]
            start_bar = note_info['start_time'] / SECONDS_PER_BAR
            duration_bars = note_info['duration'] / SECONDS_PER_BAR
            ax.broken_barh([(start_bar, duration_bars)],
                           (y - 0.4, 0.8),
                           facecolors='blue',
                           edgecolors='black',
                           linewidth=1.5)  # Thicker border to emphasise active note

        # Upcoming notes = faded
        for _, note_info in upcoming_notes.iterrows():
            note_number = note_info['note']
            note_name = midi_note_names.get(note_number, None)
            if note_name is None or note_name not in unique_notes:
                continue
            y = note_to_y[note_name]
            start_bar = note_info['start_time'] / SECONDS_PER_BAR
            duration_bars = note_info['duration'] / SECONDS_PER_BAR
            ax.broken_barh([(start_bar, duration_bars)],
                           (y - 0.4, 0.8),
                           facecolors=(0, 0, 1, 0.3),
                           edgecolors='black',
                           linewidth=0.5)

        # Add moving red line
        ax.axvline(x=current_bar, color='red', linewidth=2, linestyle='--')

        frame_filename = os.path.join(frames_dir, f"frame_{frame_idx:05d}.png")
        plt.tight_layout()
        plt.savefig(frame_filename)
        plt.close(fig)

        progress_percentage = int((frame_idx + 1) / frame_count * 100)
        progress_bar.progress(progress_percentage)

    progress_bar.empty()

    # FFmpeg: create video from frames
    temp_video_path = os.path.join(frames_dir, "temp_video.mp4")
    ffmpeg_video_cmd = [
        ffmpeg_path,
        "-y",
        "-framerate", str(FPS),
        "-i", os.path.join(frames_dir, "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        temp_video_path
    ]
    subprocess.run(ffmpeg_video_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Merge video with audio
    final_video_path = os.path.join(frames_dir, "final_video.mp4")
    ffmpeg_merge_cmd = [
        ffmpeg_path,
        "-y",
        "-i", temp_video_path,
        "-i", tmp_wav_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-strict", "experimental",
        final_video_path
    ]
    subprocess.run(ffmpeg_merge_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Read final video bytes
    with open(final_video_path, "rb") as f:
        video_bytes = f.read()

    os.remove(temp_video_path)
    os.remove(tmp_wav_path)
    os.remove(tmp_mid_path)
    shutil.rmtree(frames_dir, ignore_errors=True)

    return video_bytes

# =====================
# Actual Streamlit App
# =====================

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
    for key in ['page', 'uploaded_midi', 'selected_model', 'preview_shown', 'chosen_sample', 'generated_midi_bytes']:
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

# Page 1: Welcome
def welcome_page():
    st.image("banner.png", use_container_width=True)
    st.title("Welcome to: AI Music Assistant 🎵")
    st.markdown("""
        This interactive app allows you to upload (or select) a MIDI file and explore 
        how a AI models can generate melodic continuations. Experiment with 
        different models, view an on-screen piano roll, and download your results. 
        We hope you enjoy this taste of AI-assisted composition!
    """)

    with st.expander("View Example Videos of Input/Output"):
        tab1, tab2, tab3 = st.tabs(["Familiar Tonal Snippet", "LSTM Model Output", "Markov Model Output"])
        
        with tab1:
            st.subheader("Input MIDI: A Familiar Tonal Snippet")
            st.video("output_videos/input-3.mp4")

        with tab2:
            st.subheader("Continuation 1 - Model: LSTM")
            st.video("output_videos/output-3-lstm.mp4")

        with tab3:
            st.subheader("Continuation 2 - Model: Markov")
            st.video("output_videos/output-3-markov.mp4")

    st.markdown("---")
    st.info("When you're done viewing these examples, click the button below to continue.")
    if st.button("Continue to Instructions"):
        st.session_state.page = 'instructions'
        st.rerun()

# Page 2: Instructions
def instructions_page():
    st.title("Instructions & Overview")
    st.markdown("""
        This AI Music Assistant offers a straightforward workflow:

        1. **Select or Upload a MIDI File**  
        Choose from our provided sample MIDI files—or upload your own `.mid` file—to serve as the basis for a generated continuation.

        2. **Choose a Model**  
        Select one of two AI models:
        - **LSTM Model**: Uses a two-layer LSTM network trained for next-note prediction. It employs top‑k sampling with temperature scaling to produce smooth, coherent melodic continuations.
        - **Markov Model**: Utilizes a first‑order Markov chain built from training data. This model samples transitions between (pitch, duration) pairs to generate a continuation that retains rhythmic and melodic variability.

        3. **Visualise & Listen**  
        Instantly view a piano roll representation of the generated continuation and listen to an audio preview—no extra downloads required.

        4. **Download**  
        If you’re happy with the result, download both the generated MIDI file and a video of the piano roll in motion.

        ---
        **Model Details:**
        - **LSTM Model**: Predicts the next note based on prior context using a deep recurrent network, balancing creativity and musical coherence.
        - **Markov Model**: Generates continuations by sampling state transitions from a probabilistic chain, preserving both pitch and rhythmic information.

        ---
        *Tip: While large leaps can add excitement, they may sound jarring if overused. Try listening for a mix of smooth, stepwise motion with occasional contrasts!*
    """)

    st.info("Next step: upload or choose a MIDI file, then select a model. Explore the pop-ups and side notes for music tips!")
    st.markdown("---")
    if st.button("Proceed to Model Selection"):
        st.session_state.page = 'select_model'
        st.rerun()

# Page 3: Select Model & Upload
def select_model_page():
    st.title("Upload or Select a MIDI File, Then Choose a Model")
    st.warning("Remember: For strongly tonal examples, a V–I cadence can provide a satisfying ending!")

    input_method = st.radio("Select input method:", ["Upload MIDI", "Use Sample"])

    if input_method == "Upload MIDI":
        uploaded_file = st.file_uploader("Upload a MIDI file (.mid):", type=["mid"])
        sample_options = {}
        chosen_sample = None
    else:
        st.markdown("""
            **Sample MIDI Files Library**:
            - **(input-3)** A short excerpt using a simple scale, inspired by a familiar children's tune.
            - **(input-4)** A medium-length melody mixing different note lengths within a major scale.
            - **(input-5)** An atonal snippet with wide pitch leaps, lacking a clear key centre.
            - **(input-6)** A longer motif repeated several times, testing how models handle repetition.
        """)
        sample_options = {
            "Familiar Tonal Snippet (input-3)": "input_midis/input-3.mid",
            "Medium-Length Original Melody (input-4)": "input_midis/input-4.mid",
            "Atonal, Wide-Range Melody (input-5)": "input_midis/input-5.mid",
            "Long Repetitive Motif (input-6)": "input_midis/input-6.mid"
        }
        chosen_sample = st.selectbox("Choose a sample MIDI:", list(sample_options.keys()))
        uploaded_file = None

    st.markdown("**Select a Model:**")
    model_list = ["LSTM", "Markov"]
    chosen_model = st.selectbox("Select a model:", model_list,
                                help="Pick one of the Magenta Melody RNN models. See the details above!")

    # Track whether we've shown the preview yet
    if 'preview_shown' not in st.session_state:
        st.session_state.preview_shown = False

    # If we haven't shown the preview yet:
    if not st.session_state.preview_shown:
        if st.button("Continue to Preview"):
            # Load MIDI data
            if input_method == "Upload MIDI":
                if uploaded_file is None:
                    st.error("Please upload a MIDI file or switch to 'Use Sample'.")
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
            st.session_state.preview_shown = True
            st.rerun()
    else:
        # Preview is already shown
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

        st.info("Ready to see your 'generated' continuation? Click below!")
        if st.button("Run Model & Continue"):
            st.session_state.page = 'output'
            st.rerun()

# Page 4: Output
def output_page():
    st.title("Your Model Output")
    if 'generated_midi_bytes' not in st.session_state:
        st.session_state.generated_midi_bytes = None
    if not st.session_state.uploaded_midi:
        st.warning("No MIDI file found. Please go back and select/upload one.")
        return

    st.markdown(f"**Selected Model**: `{st.session_state.selected_model}`")
    st.markdown("""
        Below is the **generated continuation** for your MIDI file, produced by the selected model.
    """)


    if st.session_state.generated_midi_bytes is None:
        # Write the uploaded MIDI to a temporary file (this is our generation input)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as input_tmp:
            input_tmp.write(st.session_state.uploaded_midi)
            input_tmp.flush()
            input_midi_path = input_tmp.name

        # Define an output MIDI file path for the generated continuation
        output_midi_path = os.path.join(tempfile.gettempdir(), "generated_continuation.mid")

        # Call the appropriate model based on the selected option
        if st.session_state.selected_model == "LSTM":
            run_lstm_continuation(input_midi_path, output_midi_path)
        elif st.session_state.selected_model == "Markov":
            run_markov_continuation(input_midi_path, output_midi_path)
        else:
            st.error("Invalid model selected.")
            return

        # Read the generated MIDI for display and preview
        with open(output_midi_path, 'rb') as f:
            st.session_state.generated_midi_bytes = f.read()

    # We'll use generated_midi_bytes below, no matter what
    generated_midi_bytes = st.session_state.generated_midi_bytes

    # Now parse the generated MIDI into a DataFrame for the piano roll
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp:
         tmp.write(generated_midi_bytes)
         tmp.flush()
         df = parse_midi(tmp.name)


    if df.empty:
        st.warning("No valid notes found in this MIDI.")
    else:
        total_time_in_seconds = df['start_time'].max() + df['duration'].max()
        fig = create_static_piano_roll(df, total_time_in_seconds)
        st.pyplot(fig)

    try:
        wav_bytes = convert_midi_to_wav(generated_midi_bytes)
        st.markdown("### Listen to the Continuation:")
        st.audio(wav_bytes, format="audio/wav")
    except Exception as e:
        st.warning(f"Could not provide an audio preview: {e}")

    st.download_button(
        label="Download Generated MIDI",
        data=generated_midi_bytes,
        file_name="generated_continuation.mid",
        mime="audio/midi"
    )

    if st.button("Generate Piano Roll Video (.mp4)"):
        try:
            st.info("Generating piano roll video, please wait...")
            video_bytes = generate_piano_roll_video(generated_midi_bytes)
            if video_bytes:
                st.download_button(
                    label="Download Piano Roll Video",
                    data=video_bytes,
                    file_name="generated_continuation.mp4",
                    mime="video/mp4"
                )
            else:
                st.warning("Could not generate a video. Possibly no valid notes to visualise.")
        except Exception as e:
            st.error(f"Error generating piano roll video: {e}")

    st.markdown("---")
    st.info("Click below to finish or revisit your choices.")
    if st.button("Finish & Proceed"):
        st.session_state.page = 'closing'
        st.rerun()

# Page 5: Closing
def closing_page():
    if os.path.exists("closing_banner.png"):
        st.image("closing_banner.png", use_container_width=True)

    st.title("Thank You for Using the AI Music Assistant!")
    st.markdown("""
        We hope you enjoyed exploring the AI-based MIDI tool. 
        If you wish, you can select a different model below and run the same file again 
        without any additional fireworks.
    """)

    if 'closing_visited' not in st.session_state:
        st.session_state.closing_visited = True
        st.balloons()

    new_model = st.selectbox("Select a different model to rerun:", ["LSTM", "Markov"])
    # Make sure to reset generated_midi_bytes so the script runs again.
    if st.button("Rerun with New Model"):
        st.session_state.selected_model = new_model
        st.session_state.generated_midi_bytes = None
        st.session_state.page = 'output'
        st.rerun()


# Main Flow
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
