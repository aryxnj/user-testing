import os
import tempfile
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from mido import MidiFile
from midi2audio import FluidSynth
from tqdm import tqdm

# Configure these paths
SOUND_FONT_PATH = os.path.join(os.getcwd(), 'sounds', 'FluidR3_GM.sf2')
INPUT_MIDI_FILE = 'output-6-lstm.mid'
INPUT_MIDI_DIR = os.path.join(os.getcwd(), 'input_midis')
INPUT_MIDI_PATH = os.path.join(INPUT_MIDI_DIR, INPUT_MIDI_FILE)
OUTPUT_VIDEO_DIR = os.path.join(os.getcwd(), 'output_videos')
FINAL_VIDEO_PATH = os.path.join(OUTPUT_VIDEO_DIR, 'output-6-lstm.mp4')
FPS = 30
BPM = 130
BEATS_PER_BAR = 4
SECONDS_PER_BEAT = 60 / BPM
SECONDS_PER_BAR = SECONDS_PER_BEAT * BEATS_PER_BAR

# Initialize FluidSynth with the specified path
fs = FluidSynth(sound_font=SOUND_FONT_PATH)


def convert_midi_to_audio(midi_file):
    """
    Convert a MIDI file to WAV audio using FluidSynth.
    Returns the path to the generated WAV file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        fs.midi_to_audio(midi_file, tmp_audio.name)
        return tmp_audio.name

def parse_midi(file_path):
    """
    Parse a MIDI file and extract note information.
    Returns a pandas DataFrame with columns: note, velocity, start_time, duration.
    """
    midi = MidiFile(file_path)
    notes = []
    tempo = 500000  # Default tempo (120 BPM)
    ticks_per_beat = midi.ticks_per_beat
    track_times = [0] * len(midi.tracks)  # Keep track of time per track

    for i, track in enumerate(midi.tracks):
        for msg in track:
            track_times[i] += msg.time  # Accumulate time for the current track
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            elif msg.type == 'note_on' and msg.velocity > 0:
                note_start = track_times[i] * tempo / ticks_per_beat / 1_000_000  # Convert ticks to seconds
                notes.append({
                    'note': msg.note,
                    'velocity': msg.velocity,
                    'start_time': note_start,
                })
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                for note in reversed(notes):
                    if note['note'] == msg.note and 'duration' not in note:
                        note_end = track_times[i] * tempo / ticks_per_beat / 1_000_000
                        note['duration'] = note_end - note['start_time']
                        break

    # Remove notes without end times
    notes = [note for note in notes if 'duration' in note]

    # Adjust all start times to begin at 0
    if notes:
        min_start_time = min(note['start_time'] for note in notes)
        for note in notes:
            note['start_time'] -= min_start_time

    return pd.DataFrame(notes)

def generate_midi_note_names():
    """
    Generate a mapping from MIDI note numbers to note names.
    Returns a dictionary {note_number: note_name}.
    """
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                 'F#', 'G', 'G#', 'A', 'A#', 'B']
    midi_note_names = {}
    for i in range(21, 109):  # Standard piano range: A0 (21) to C8 (108)
        octave = (i // 12) - 1
        note = note_names[i % 12]
        midi_note_names[i] = f"{note}{octave}"
    return midi_note_names

def create_video_frames(data, total_time_in_seconds, fps=30):
    """
    Create image frames representing the piano roll visualization.
    Saves frames as images in a temporary directory.
    Returns the path to the directory containing frames.
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
    
    # Create temporary directory for frames
    temp_frames_dir = tempfile.mkdtemp()
    
    # Calculate frame count based on total time in seconds
    frame_count = int(total_time_in_seconds * fps)
    
    # Precompute active, past, and upcoming notes per frame
    data_sorted = data.sort_values('start_time').reset_index(drop=True)
    current_index = 0
    active_notes = []
    past_notes = []
    
    for frame_idx in tqdm(range(frame_count), desc="Generating frames"):
        current_time = frame_idx / fps  # in seconds
        current_bar = current_time / SECONDS_PER_BAR

        # Add new notes to active_notes
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

        # Move notes that have ended from active_notes to past_notes
        for note in active_notes[:]:
            if note['end_time'] <= current_time:
                past_notes.append(note)
                active_notes.remove(note)

        # Identify all upcoming notes (no upper limit)
        upcoming_notes = data_sorted[
            (data_sorted['start_time'] > current_time)
        ]

        # Plotting
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_ylim(-1, len(unique_notes))
        ax.set_xlim(0, total_time_in_seconds / SECONDS_PER_BAR)  # X-axis in bars
        ax.set_xlabel("Bars")
        ax.set_ylabel("Notes")
        ax.set_title("FL Studio-Style Piano Roll")
        ax.set_yticks(range(len(unique_notes)))
        ax.set_yticklabels(unique_notes)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Plot past notes as faded blue boxes with borders
        for note_info in past_notes:
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
                          edgecolors='black',          # Black border
                          linewidth=0.5)                # Border width

        # Plot active notes as solid blue boxes
        for note_info in active_notes:
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
                          facecolors='blue')  # Solid blue

        # Plot upcoming notes as faded blue boxes with borders
        for _, note in upcoming_notes.iterrows():
            note_number = note['note']
            note_name = midi_note_names.get(note_number, None)
            if note_name is None or note_name not in unique_notes:
                continue  # Skip notes outside the filtered range
            y = note_to_y[note_name]
            # Calculate start_bar and duration_bars
            start_bar = note['start_time'] / SECONDS_PER_BAR
            duration_bars = note['duration'] / SECONDS_PER_BAR
            ax.broken_barh([(start_bar, duration_bars)],
                          (y - 0.4, 0.8),
                          facecolors=(0, 0, 1, 0.3),  # Faded blue with alpha=0.3
                          edgecolors='black',          # Black border
                          linewidth=0.5)                # Border width

        # Add the moving vertical line indicating current bar
        ax.axvline(x=current_bar, color='red', linewidth=2, linestyle='--')

        # Save the current frame as an image
        frame_filename = os.path.join(temp_frames_dir, f"frame_{frame_idx:05d}.png")
        plt.tight_layout()
        plt.savefig(frame_filename)
        plt.close(fig)

    return temp_frames_dir

def create_video_ffmpeg(frames_dir, audio_path, final_video_path, fps=30):
    """
    Create a video from image frames using FFmpeg and add audio.
    """
    # FFmpeg command to create video from frames
    video_command = [
        'ffmpeg',
        '-y',  # Overwrite output files without asking
        '-framerate', str(fps),
        '-i', os.path.join(frames_dir, 'frame_%05d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        os.path.join(frames_dir, 'temp_video.mp4')
    ]

    try:
        print("  Creating video from frames using FFmpeg...")
        result = subprocess.run(video_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("  Video creation from frames completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"  FFmpeg error during video creation:\n{e.stderr.decode()}")
        return

    # FFmpeg command to merge video and audio
    merge_command = [
        'ffmpeg',
        '-y',
        '-i', os.path.join(frames_dir, 'temp_video.mp4'),
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-strict', 'experimental',
        final_video_path
    ]

    try:
        print("  Merging video and audio using FFmpeg...")
        result = subprocess.run(merge_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"  Video saved to {final_video_path}\n")
    except subprocess.CalledProcessError as e:
        print(f"  FFmpeg error during merging:\n{e.stderr.decode()}")
        print("  Failed to merge video and audio.")

    # Cleanup temporary video file
    temp_video_path = os.path.join(frames_dir, 'temp_video.mp4')
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)

def process_midi_file(midi_file, input_path, output_dir):
    """
    Process a single MIDI file: convert to audio, generate frames, and create video.
    """
    final_video_path = os.path.join(output_dir, f"{os.path.splitext(midi_file)[0]}.mp4")

    print(f"Processing {midi_file}...")

    # Parse MIDI for visualization
    midi_data = parse_midi(input_path)
    if midi_data.empty:
        print(f"No playable notes found in {midi_file}. Skipping.")
        return

    # Calculate total time in seconds
    total_time_in_seconds = midi_data['start_time'].max() + midi_data['duration'].max()

    # Convert MIDI to audio
    print("  Converting MIDI to audio...")
    audio_path = convert_midi_to_audio(input_path)

    # Generate frames and save as images
    print("  Generating video frames...")
    frames_dir = create_video_frames(midi_data, total_time_in_seconds, fps=FPS)

    # Create video from frames and merge with audio using FFmpeg
    create_video_ffmpeg(frames_dir, audio_path, final_video_path, fps=FPS)

    # Cleanup temporary frames directory and audio file
    if os.path.exists(audio_path):
        os.remove(audio_path)
    if os.path.exists(frames_dir):
        import shutil
        shutil.rmtree(frames_dir)

def main():
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_VIDEO_DIR):
        os.makedirs(OUTPUT_VIDEO_DIR)

    # Check if the input MIDI file exists
    if not os.path.isfile(INPUT_MIDI_PATH):
        print(f"Input MIDI file '{INPUT_MIDI_FILE}' not found in the current directory.")
        return

    # Process the MIDI file
    try:
        process_midi_file(INPUT_MIDI_FILE, INPUT_MIDI_PATH, OUTPUT_VIDEO_DIR)
    except Exception as e:
        print(f"Error processing {INPUT_MIDI_FILE}: {e}\n")

    print("All MIDI files have been processed.")

if __name__ == "__main__":
    main()
