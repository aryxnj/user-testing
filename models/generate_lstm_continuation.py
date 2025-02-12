#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
import pretty_midi

##############################
# Configuration & Constants  #
##############################
MODEL_PATH = "models/lstm_model.h5"  # Update the path as needed
# The following defaults can be overridden by function parameters
DEFAULT_NUM_GENERATION_STEPS = 16  # How many new notes to generate
DEFAULT_TEMPERATURE = 0.6          # Lower => more predictable
DEFAULT_TOP_K = 5                  # Restrict sampling to top-5 tokens
DEFAULT_MAX_INTERVAL = 12          # Prevent leaps larger than one octave

MIN_PITCH = 0
MAX_PITCH = 127

########################
# 1. Load the Model    #
########################
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

########################################################
# 2. MIDI -> Note Sequence in [0..127]                 #
########################################################
def midi_to_note_sequence(midi_path):
    """
    Extract the main monophonic melody as pitches in [0..127].
    Durations are ignored, so we only gather pitch values.
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    
    # Find the first non-drum instrument with notes
    instrument = None
    for inst in pm.instruments:
        if not inst.is_drum and len(inst.notes) > 0:
            instrument = inst
            break
    if instrument is None:
        print(f"Could not find any monophonic track in {midi_path}.")
        return []
    
    # Sort notes by start time
    instrument.notes.sort(key=lambda n: n.start)
    
    note_sequence = []
    for note in instrument.notes:
        pitch_val = np.clip(note.pitch, MIN_PITCH, MAX_PITCH)
        note_sequence.append(pitch_val)
    
    return note_sequence

########################################################
# 3. Top-k Sampling + Clamp to Prevent Big Leaps       #
########################################################
def top_k_sample(logits, k=DEFAULT_TOP_K):
    """
    Restrict to the top-k most likely tokens, then sample from that subset.
    Returns a Python int for the chosen pitch index.
    """
    logits = tf.reshape(logits, [-1])  # shape: (vocab_size,)

    # Extract top-k scores and their indices
    values, indices = tf.math.top_k(logits, k=k)

    # Convert top-k values to log probabilities
    restricted_logits = tf.nn.log_softmax(values)

    # Sample from the top-k distribution
    next_token = tf.random.categorical(
        tf.expand_dims(restricted_logits, 0),
        num_samples=1
    )
    # next_token is a 1D Tensor, convert to Python int
    next_token_index = int(tf.squeeze(next_token, axis=-1).numpy())
    
    # Use next_token_index to get the actual pitch from 'indices'
    predicted_id = indices[next_token_index]
    
    # predicted_id is still a Tensor; convert to Python int
    return int(predicted_id.numpy())

def clamp_pitch(prev_pitch, new_pitch, max_interval=DEFAULT_MAX_INTERVAL):
    """
    Clamp 'new_pitch' so it can't jump more than 'max_interval' semitones 
    above or below 'prev_pitch', staying within [0..127].
    """
    lower_bound = max(prev_pitch - max_interval, MIN_PITCH)
    upper_bound = min(prev_pitch + max_interval, MAX_PITCH)
    return int(np.clip(new_pitch, lower_bound, upper_bound))

###################################################################
# 4. Generate Additional Notes by Autoregressive Sampling         #
###################################################################
def generate_continuation(
    model,
    seed_sequence,
    num_steps=DEFAULT_NUM_GENERATION_STEPS,
    temperature=DEFAULT_TEMPERATURE
):
    """
    Predict the next 'num_steps' notes from 'seed_sequence' [0..127],
    with top-k sampling and pitch clamping to avoid big leaps.
    """
    generated = list(seed_sequence)  # Start with the original seed
    input_seq = tf.expand_dims(generated, 0)  # shape: (1, length)

    for _ in range(num_steps):
        # model output shape: (1, current_length, VOCAB_SIZE)
        logits = model(input_seq)[:, -1, :]  # last time step's logits
        logits = logits / temperature

        # Sample from the top-k distribution
        sampled_pitch = top_k_sample(logits, k=DEFAULT_TOP_K)

        # Clamp pitch to prevent large leaps from the last note
        if generated:
            last_pitch = generated[-1]
        else:
            last_pitch = 60  # default to middle C if empty
        clamped_pitch = clamp_pitch(last_pitch, sampled_pitch, DEFAULT_MAX_INTERVAL)

        generated.append(clamped_pitch)
        
        # Update the input_seq
        input_seq = tf.expand_dims(generated, 0)

    return generated

##################################################
# 5. Convert the Generated Sequence back to MIDI #
##################################################
def note_sequence_to_midi(note_sequence, output_path):
    """
    Render pitches in [0..127] into a monophonic MIDI with fixed 0.5s note durations.
    """
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Piano
    pm.instruments.append(instrument)
    
    current_time = 0.0
    NOTE_DURATION = 0.5
    VELOCITY = 100
    
    for pitch in note_sequence:
        note = pretty_midi.Note(
            velocity=VELOCITY,
            pitch=pitch,
            start=current_time,
            end=current_time + NOTE_DURATION
        )
        instrument.notes.append(note)
        current_time += NOTE_DURATION
    
    pm.write(output_path)
    print(f"Saved generated MIDI to {output_path}")

##############################################
# 6. Main function to run LSTM continuation  #
##############################################
def run_lstm_continuation(input_midi_path, output_midi_path):
    """
    Given an input MIDI file path, generate a continuation using the LSTM model
    and save the output MIDI to the specified output path.
    Returns the output MIDI path.
    """
    # 1. Convert input MIDI -> note sequence
    seed_notes = midi_to_note_sequence(input_midi_path)
    if not seed_notes:
        print("No seed notes found; exiting.")
        return None
    
    print(f"Seed sequence length: {len(seed_notes)} notes.")
    
    # 2. Generate a continuation
    extended_sequence = generate_continuation(
        model,
        seed_notes,
        num_steps=DEFAULT_NUM_GENERATION_STEPS,
        temperature=DEFAULT_TEMPERATURE
    )
    
    # 3. Convert to MIDI and save
    note_sequence_to_midi(extended_sequence, output_midi_path)
    return output_midi_path

if __name__ == "__main__":
    # For command-line testing, use default paths
    DEFAULT_INPUT_MIDI_PATH = "input_midis/input-3.mid"
    DEFAULT_OUTPUT_MIDI_PATH = "output_midis/output-3-generated.mid"
    run_lstm_continuation(DEFAULT_INPUT_MIDI_PATH, DEFAULT_OUTPUT_MIDI_PATH)
