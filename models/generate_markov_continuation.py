import pickle
import random
import mido
from collections import defaultdict

def load_markov_model(model_path):
    """
    Load the pre-trained Markov chain transition probabilities from a pickle file.
    """
    with open(model_path, 'rb') as f:
        transition_probs = pickle.load(f)
    return transition_probs

def parse_input_midi(midi_path):
    """
    Parse a monophonic MIDI file to extract a sequence of [pitch, duration] pairs,
    searching for the first track that contains any note messages.
    """
    midi_data = mido.MidiFile(midi_path)

    # Find the first track with note_on or note_off messages
    track = None
    for t in midi_data.tracks:
        if any(msg.type in ("note_on", "note_off") for msg in t):
            track = t
            break

    if not track:
        print("No MIDI tracks with note data found. Returning an empty melody.")
        return []

    ticks_per_unit = midi_data.ticks_per_beat // 2 if midi_data.ticks_per_beat else 120
    current_time = 0
    note_on_times = {}
    melody = []

    for msg in track:
        current_time += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            # Start of a note
            note_on_times[msg.note] = current_time
        elif (msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)):
            # End of a note
            pitch = msg.note
            if pitch in note_on_times:
                on_time = note_on_times[pitch]
                off_time = current_time
                duration_in_ticks = off_time - on_time
                duration_units = max(1, round(duration_in_ticks / ticks_per_unit))
                melody.append([pitch, duration_units])
                del note_on_times[pitch]

    return melody

def sample_next_state(current_state, transition_probs):
    """
    Given the current (pitch, duration) and the transition probabilities,
    sample a next state (pitch, duration) according to the learned distribution.
    """
    if current_state in transition_probs:
        next_states = list(transition_probs[current_state].keys())
        probabilities = list(transition_probs[current_state].values())
        return random.choices(next_states, weights=probabilities, k=1)[0]
    else:
        # Fallback: pick a random state from all keys
        all_states = list(transition_probs.keys())
        return random.choice(all_states)

def generate_continuation(seed_melody, transition_probs, num_notes=16):
    """
    Generate a continuation of length num_notes, starting from the last state
    of the seed_melody.
    
    Returns a list of (pitch, duration) pairs.
    """
    # Convert the seed to tuples to match dictionary keys
    seed_tuples = [tuple(note) for note in seed_melody]
    
    if not seed_tuples:
        # If the seed is empty, pick a random state from the chain
        current_state = random.choice(list(transition_probs.keys()))
    else:
        # Otherwise, start from the final note of the seed
        current_state = seed_tuples[-1]
    
    generated = []
    for _ in range(num_notes):
        next_state = sample_next_state(current_state, transition_probs)
        generated.append(next_state)
        current_state = next_state
    return generated

def build_midi_from_sequence(sequence, ticks_per_beat=480):
    """
    Convert a list of (pitch, duration) pairs into a monophonic MIDI file.

    We assume that 'duration' is in 'units', and we choose how many ticks each
    unit should represent. For example, if ticks_per_beat=480 and we set
    ticks_per_unit to 240, then:
       duration 1 => 240 ticks
       duration 2 => 480 ticks, etc.
    """
    midi_file = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi_file.tracks.append(track)

    # Decide how many ticks for each duration unit
    ticks_per_unit = ticks_per_beat // 2  # e.g., 1 => half a beat
    
    for (pitch, dur) in sequence:
        # Note on at time = 0
        track.append(mido.Message('note_on', note=pitch, velocity=64, time=0))
        # Note off after duration
        track.append(mido.Message('note_off', note=pitch, velocity=64, 
                                  time=dur * ticks_per_unit))
    
    return midi_file

##############################################
# 6. Main function to run Markov continuation #
##############################################
def run_markov_continuation(input_midi_path, output_midi_path):
    """
    Given an input MIDI file path, generate a continuation using the Markov model
    and save the output MIDI to the specified output path.
    Returns the output MIDI path.
    """
    # 1. Load the pre-trained Markov model
    model_path = "models/markov_model.pkl"
    transition_probs = load_markov_model(model_path)
    
    # 2. Parse the input MIDI (monophonic) into a seed melody
    seed_melody = parse_input_midi(input_midi_path)
    if not seed_melody:
        print("Warning: The input MIDI might be empty or invalid. Proceeding with a random seed.")
    
    # 3. Generate a continuation (16 notes by default)
    generated_notes = generate_continuation(seed_melody, transition_probs, num_notes=16)
    
    # 4. Combine the seed and the generated sequence
    full_sequence = seed_melody + [list(note) for note in generated_notes]
    
    # 5. Convert to a MIDI and save
    midi_output = build_midi_from_sequence(full_sequence, ticks_per_beat=480)
    midi_output.save(output_midi_path)
    print(f"Generated continuation saved to {output_midi_path}")
    return output_midi_path

if __name__ == "__main__":
    # For command-line testing, use default paths
    DEFAULT_INPUT_MIDI_PATH = "input_midis/input-3.mid"
    DEFAULT_OUTPUT_MIDI_PATH = "output_midis/generated_continuation.mid"
    run_markov_continuation(DEFAULT_INPUT_MIDI_PATH, DEFAULT_OUTPUT_MIDI_PATH)
