import streamlit as st
import os
import io
import tempfile
import shutil
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from mido import MidiFile
from midi2audio import FluidSynth
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

# ================== Configuration (from the script) ==================
SOUND_FONT_PATH = os.path.join(os.getcwd(), 'sounds', 'FluidR3_GM.sf2')  # Path to SoundFont
FLUIDSYNTH_PATH = 'C:\\Program Files\\FluidSynth\\fluidsynth.exe'       # Example path on Windows
fs = FluidSynth(sound_font=SOUND_FONT_PATH)

# Playback settings
FPS = 30
BPM = 130
BEATS_PER_BAR = 4
SECONDS_PER_BEAT = 60 / BPM
SECONDS_PER_BAR = SECONDS_PER_BEAT * BEATS_PER_BAR

# =========== Functions Copied Exactly from Provided Script ===========

def convert_midi_to_audio(midi_file_path):
    """
    Convert a MIDI file (on disk) to WAV audio using FluidSynth.
    Returns the path to the generated WAV file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        fs.midi_to_audio(midi_file_path, tmp_audio.name)
        return tmp_audio.name

def parse_midi(file_path):
    """
    Parse a MIDI file and extract note information.
    Returns a pandas DataFrame with columns: note, velocity, start_time, duration.
    """
    midi = MidiFile(file_path)
    notes = []
    tempo = 500000  # Default tempo
    ticks_per_beat = midi.ticks_per_beat
    track_times = [0] * len(midi.tracks)  # Keep track of time per track

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
                for note in reversed(notes):
                    if note['note'] == msg.note and 'duration' not in note:
                        note_end = track_times[i] * tempo / ticks_per_beat / 1_000_000
                        note['duration'] = note_end - note['start_time']
                        break

    # Remove notes without end times
    notes = [n for n in notes if 'duration' in n]

    # Adjust all start times so earliest note begins at 0
    if notes:
        min_start_time = min(n['start_time'] for n in notes)
        for n in notes:
            n['start_time'] -= min_start_time

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

def single_frame_piano_roll_plot(data, current_time=0):
    """
    This function replicates the logic from create_video_frames(...) for a SINGLE frame.
    We show 'past', 'active', and 'upcoming' notes based on current_time.
    Includes the vertical red line.
    """

    midi_note_names = generate_midi_note_names()

    # Define full piano note range from A0 to C8
    full_range_notes = [midi_note_names[i] for i in range(21, 109)]
    note_to_y_full = {note: idx for idx, note in enumerate(full_range_notes)}

    # Map the DataFrame's note column to note names
    data['note_name'] = data['note'].map(midi_note_names)
    data = data.dropna(subset=['note_name'])

    if data.empty:
        # Return a figure stating no notes
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No notes to display.", ha='center', va='center', fontsize=14)
        return fig

    # Identify unique octaves actually used
    played_notes = data['note'].unique()
    played_octaves = sorted(set((pn // 12) - 1 for pn in played_notes if 21 <= pn <= 108))

    # Rebuild an abbreviated note list only for the played octaves
    filtered_notes = []
    for octave in played_octaves:
        for note in ['C', 'C#', 'D', 'D#', 'E', 'F',
                     'F#', 'G', 'G#', 'A', 'A#', 'B']:
            note_index = ['C', 'C#', 'D', 'D#', 'E', 'F',
                          'F#', 'G', 'G#', 'A', 'A#', 'B'].index(note)
            midi_num = (octave + 1) * 12 + note_index
            if 21 <= midi_num <= 108:
                filtered_notes.append(f"{note}{octave}")

    filtered_note_to_y = {note: idx for idx, note in enumerate(filtered_notes)}

    total_time_in_seconds = (data['start_time'] + data['duration']).max()

    # Distinguish 'past', 'active', and 'upcoming' notes at current_time
    # We'll define a small epsilon to handle floating comparisons
    epsilon = 1e-9
    past_data = data[data['start_time'] + data['duration'] < current_time - epsilon]
    active_data = data[(data['start_time'] <= current_time + epsilon) &
                       (data['start_time'] + data['duration'] >= current_time - epsilon)]
    upcoming_data = data[data['start_time'] > current_time + epsilon]

    # Plot
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_ylim(-1, len(filtered_notes))
    # We show x-axis in 'bars' just as in original code
    max_bars = total_time_in_seconds / SECONDS_PER_BAR if total_time_in_seconds > 0 else 1
    ax.set_xlim(0, max_bars)
    ax.set_xlabel("Bars")
    ax.set_ylabel("Notes")
    ax.set_title("Piano Roll (Single Frame)")
    ax.set_yticks(range(len(filtered_notes)))
    ax.set_yticklabels(filtered_notes)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Helper to convert seconds -> bars
    def sec_to_bar(t_sec):
        return t_sec / SECONDS_PER_BAR

    # Past notes in faded blue
    for _, row in past_data.iterrows():
        note_name = row['note_name']
        if note_name not in filtered_note_to_y:
            continue
        start_bar = sec_to_bar(row['start_time'])
        duration_bars = sec_to_bar(row['duration'])
        y = filtered_note_to_y[note_name]
        ax.broken_barh([(start_bar, duration_bars)], (y - 0.4, 0.8),
                       facecolors=(0, 0, 1, 0.3),
                       edgecolors='black', linewidth=0.5)

    # Active notes in solid blue
    for _, row in active_data.iterrows():
        note_name = row['note_name']
        if note_name not in filtered_note_to_y:
            continue
        start_bar = sec_to_bar(row['start_time'])
        duration_bars = sec_to_bar(row['duration'])
        y = filtered_note_to_y[note_name]
        ax.broken_barh([(start_bar, duration_bars)], (y - 0.4, 0.8),
                       facecolors='blue')

    # Upcoming notes in faded blue
    for _, row in upcoming_data.iterrows():
        note_name = row['note_name']
        if note_name not in filtered_note_to_y:
            continue
        start_bar = sec_to_bar(row['start_time'])
        duration_bars = sec_to_bar(row['duration'])
        y = filtered_note_to_y[note_name]
        ax.broken_barh([(start_bar, duration_bars)], (y - 0.4, 0.8),
                       facecolors=(0, 0, 1, 0.3),
                       edgecolors='black', linewidth=0.5)

    # Red vertical line at the current bar
    current_bar = sec_to_bar(current_time)
    ax.axvline(x=current_bar, color='red', linewidth=2, linestyle='--')

    plt.tight_layout()
    return fig

def convert_midi_bytes_to_wav(midi_bytes: bytes) -> bytes:
    """
    Convert raw MIDI data in memory to WAV data in memory.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp_midi:
        tmp_midi.write(midi_bytes)
        tmp_midi_path = tmp_midi.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        tmp_wav_path = tmp_wav.name

    fs.midi_to_audio(tmp_midi_path, tmp_wav_path)

    with open(tmp_wav_path, 'rb') as f:
        wav_data = f.read()

    # Cleanup
    os.remove(tmp_midi_path)
    os.remove(tmp_wav_path)
    return wav_data

def generate_piano_roll_video_placeholder(midi_bytes: bytes) -> bytes:
    """
    Placeholder for generating a piano roll video. We won't actually do it here,
    but in a real scenario, you'd replicate create_video_frames -> create_video_ffmpeg.
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
if 'method_chosen' not in st.session_state:
    st.session_state.method_chosen = None       # "upload" or "sample"
if 'show_preview' not in st.session_state:
    st.session_state.show_preview = False       # Whether to display the piano roll & audio

def reset_session():
    for key in ['page', 'uploaded_midi', 'selected_model', 'method_chosen', 'show_preview']:
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
    st.image("banner.png", use_container_width=True)
    st.markdown("""
        This application enables you to upload or select a MIDI snippet, then choose a 
        simple AI model to 'continue' that snippet (currently returning the same file, for now).
    """)
    st.markdown("---")
    if st.button("Continue to Instructions"):
        st.session_state.page = 'instructions'
        st.rerun()

# ================== Page 2: Instructions ==================
def instructions_page():
    st.title("Instructions & Overview")
    st.markdown("""
        1. **Choose how to provide your MIDI**: Either upload your own or pick from one of the provided sample snippets. 
           You cannot do both at once—it's one or the other.
        2. **Preview**: Once you continue to preview, you will see a single-frame "piano roll" showing 
           what's already happened, what's currently playing, and what's upcoming, plus an audio player so you 
           can listen to your snippet.
        3. **Run Model**: Clicking "Run Model & Continue" will (for now) simply pass you on to the output page, 
           where you can download or optionally generate a placeholder video of the piano roll.
        4. **Repeat or Finish**: At the final page, you can choose a different model or end your session.
    """)
    st.markdown("---")
    if st.button("Proceed to Model Selection"):
        st.session_state.page = 'select_model'
        st.rerun()

# ================== Page 3: Select Model & Upload ==================
def select_model_page():
    st.title("Select or Upload a MIDI File, Then Choose a Model")

    # Provide radio buttons to choose method
    st.markdown("### Step 1: Choose how you'd like to provide your MIDI:")
    method = st.radio(
        "Select an option:",
        ["Upload my own MIDI file", "Use a sample MIDI file"],
        index=0 if st.session_state.method_chosen != "sample" else 1
    )
    st.session_state.method_chosen = "upload" if method == "Upload my own MIDI file" else "sample"

    # If user picks "upload," disable sample selection
    # If user picks "sample," disable file uploader
    if st.session_state.method_chosen == "upload":
        uploaded_file = st.file_uploader(
            "Upload a MIDI file (.mid):",
            type=["mid"],
            disabled=False
        )
        # Grey out sample selection
        sample_options = {
            "Familiar Tonal Snippet": ("input_midis", "input-3.mid"),
            "Medium-Length Original Melody": ("input_midis", "input-4.mid"),
            "Atonal, Wide-Range Melody": ("input_midis", "input-5.mid"),
            "Long Repetitive Motif": ("input_midis", "input-6.mid")
        }
        st.selectbox(
            "Sample MIDI options are disabled when uploading your own:",
            list(sample_options.keys()),
            disabled=True
        )
    else:
        # Grey out the file uploader
        st.file_uploader(
            "Uploader disabled when using a sample:",
            type=["mid"],
            disabled=True
        )
        sample_options = {
            "Familiar Tonal Snippet": ("input_midis", "input-3.mid"),
            "Medium-Length Original Melody": ("input_midis", "input-4.mid"),
            "Atonal, Wide-Range Melody": ("input_midis", "input-5.mid"),
            "Long Repetitive Motif": ("input_midis", "input-6.mid")
        }
        chosen_sample_name = st.selectbox(
            "Pick a sample MIDI from our library:",
            ["(Select a sample)"] + list(sample_options.keys())
        )
        uploaded_file = None

    st.markdown("---")
    st.markdown("### Step 2: Choose a Model:")
    model_list = ["basic", "lookback", "attention", "mono"]
    chosen_model = st.selectbox("Select a model:", model_list)

    st.markdown("---")

    # If user has not yet pressed "Continue to Preview", show the button. 
    # Once pressed, we show the piano roll & audio
    if not st.session_state.show_preview:
        if st.button("Continue to Preview"):
            # Validate input
            if st.session_state.method_chosen == "upload":
                # Need an uploaded file
                if not uploaded_file:
                    st.error("Please upload a MIDI file first.")
                    return
                # Store it
                st.session_state.uploaded_midi = uploaded_file.getvalue()
            else:
                # Must have a chosen sample
                if not chosen_sample_name or chosen_sample_name == "(Select a sample)":
                    st.error("Please select a sample MIDI.")
                    return
                sample_dir, sample_file = sample_options[chosen_sample_name]
                sample_path = os.path.join(sample_dir, sample_file)
                try:
                    with open(sample_path, "rb") as f:
                        st.session_state.uploaded_midi = f.read()
                except Exception as e:
                    st.error(f"Could not load sample: {e}")
                    return

            st.session_state.selected_model = chosen_model
            st.session_state.show_preview = True
            st.rerun()
    else:
        # We are in "preview" mode
        st.success(f"Loaded MIDI successfully! (Model: {st.session_state.selected_model})")
        
        # Show single-frame piano roll
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp:
                tmp.write(st.session_state.uploaded_midi)
                tmp.flush()
                df = parse_midi(tmp.name)
        except Exception as e:
            st.error(f"Unable to parse MIDI: {e}")
            return

        if df.empty:
            st.warning("No notes in this MIDI or parsing failed.")
        else:
            # Let's pick a 'current_time' to be somewhere in the middle for demonstration
            # e.g. halfway through total length or near the start
            total_time = (df['start_time'] + df['duration']).max()
            current_time = min(1.0, total_time / 2.0)  # for demonstration
            fig = single_frame_piano_roll_plot(df, current_time=current_time)
            st.pyplot(fig)

        # Provide an audio player
        try:
            wav_data = convert_midi_bytes_to_wav(st.session_state.uploaded_midi)
            st.markdown("### Listen to Your MIDI:")
            st.audio(wav_data, format="audio/wav")
        except Exception as e:
            st.warning(f"Could not create audio preview: {e}")

        st.markdown("---")
        # Final button to actually "run the model"
        if st.button("Run Model & Continue"):
            # In reality, we'd do something with the model. For now, just proceed to the output page.
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
        Below is the 'continuation' for your MIDI file. Currently, our system just returns the same file.
    """)

    # Let's parse the data again
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp:
            tmp.write(st.session_state.uploaded_midi)
            tmp.flush()
            df = parse_midi(tmp.name)
    except Exception as e:
        st.error(f"Could not parse MIDI in output page: {e}")
        return

    if df.empty:
        st.warning("No valid notes found in the final output.")
    else:
        # Show a single-frame at the start or middle
        total_time = (df['start_time'] + df['duration']).max()
        # For fun, let's place the current_time near the start
        current_time = min(0.5, total_time)
        fig = single_frame_piano_roll_plot(df, current_time=current_time)
        st.pyplot(fig)

    # Provide an audio preview
    try:
        wav_data = convert_midi_bytes_to_wav(st.session_state.uploaded_midi)
        st.markdown("### Listen to the Continuation:")
        st.audio(wav_data, format="audio/wav")
    except Exception as e:
        st.warning(f"Unable to generate audio preview: {e}")

    # MIDI download
    st.download_button(
        label="Download Generated MIDI",
        data=st.session_state.uploaded_midi,
        file_name="generated_continuation.mid",
        mime="audio/midi"
    )

    # Provide placeholder for generating a video
    if st.button("Generate Piano Roll Video (.mp4)"):
        try:
            video_bytes = generate_piano_roll_video_placeholder(st.session_state.uploaded_midi)
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
    # Show a closing banner if you have one
    if os.path.exists("closing_banner.png"):
        st.image("closing_banner.png", use_container_width=True)

    st.title("Thank You for Using the AI Music Assistant!")
    st.markdown("""
        You can select a different model below and re-run if you wish.
        Otherwise, feel free to close the application.
    """)

    if 'closing_visited' not in st.session_state:
        st.session_state.closing_visited = True
        st.balloons()

    new_model = st.selectbox("Choose a different model:", ["basic", "lookback", "attention", "mono"])
    if st.button("Rerun with New Model"):
        st.session_state.selected_model = new_model
        st.session_state.page = 'output'
        # We skip balloons on this re-run
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
