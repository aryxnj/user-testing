# AI Music Assistant: Intelligent Music Prediction Using Deep Learning

Welcome to the repository for a final year computer science dissertation project on deep learning for music continuation. This repository contains three Python scripts:

1. **Piano Roll Video Generation**: Produces piano roll video visualisations from MIDI files.  
2. **User Testing Front End**: A Streamlit-based application for gathering user ratings of different model continuations.  
3. **AI Music Assistant Front End**: A planned Streamlit application, providing an interface to upload or create a MIDI file, select a model, generate a continuation, and optionally listen to or download the output.

---

## Contents

1. [Overview](#overview)  
2. [Prerequisites](#prerequisites)  
3. [Scripts](#scripts)  
   - [Piano Roll Video Generation](#piano-roll-video-generation)  
   - [User Testing Front End](#user-testing-front-end)  
   - [AI Music Assistant Front End](#ai-music-assistant-front-end)  
4. [Usage](#usage)  
5. [Licence](#licence)

---

## Overview

This repository showcases an AI Music Assistant developed for a dissertation project. The system is designed to generate musical continuations for given MIDI inputs using deep learning models. The included scripts cover:

- Automated piano roll video generation for visualisation and demonstration.  
- A user testing platform, enabling the collection of ratings and feedback on multiple models’ output.  
- A forthcoming assistant application that will allow users to upload MIDI files, create new melodies, apply different models, and download or play back the resulting continuation.

---

## Prerequisites

- **Python 3.7+** (tested up to Python 3.10)
- [pip](https://pypi.org/project/pip/) or similar for managing packages

You will also need the following libraries (install them via `pip install <package>` or similar):

- `mido`  
- `midi2audio`  
- `pandas`  
- `matplotlib`  
- `streamlit`  
- `sqlalchemy`  
- `psycopg2` (if using PostgreSQL)  
- `tqdm`  
- `requests` (if required for your usage)

Ensure you have [FluidSynth](https://github.com/FluidSynth/fluidsynth) installed and a compatible SoundFont file (for instance, `FluidR3_GM.sf2`). Update the `SOUND_FONT_PATH` in the scripts to match your local setup.

---

## Scripts

### Piano Roll Video Generation
- **File**: piano_roll_video.py  
- **Purpose**:  
  - Parses a specified MIDI file to extract note information.  
  - Converts the MIDI data into audio using FluidSynth.  
  - Generates piano roll video frames with bars and note placements.  
  - Creates a final `.mp4` file by merging frames with the audio track via FFmpeg.  
- **Key Features**:  
  - Adjustable frames per second and tempo.  
  - Automatic handling of note start times and durations.  
  - Configurable directories and file paths.

### User Testing Front End
- **File**: app.py
- **Purpose**:  
  - Provides a Streamlit-based interface for participants to listen to and rate different model continuations of various input melodies.  
  - Tracks user background information, rating criteria, and feedback.  
  - Stores collected responses (optionally in a PostgreSQL database).  
- **Key Features**:  
  - Dynamically loads input and output `.mp4` files.  
  - Guides the user through a step-by-step rating process.  
  - Saves feedback through a form submission process.

### AI Music Assistant Front End
- **File**: app2.py
- **Purpose**:  
  - Offers a Streamlit-based interface for uploading or creating MIDI files with a piano roll, selecting a deep learning model, generating a continuation, and exploring or downloading the result.  
  - Intended to be an evolution of the user testing front end, providing more direct interaction with the models and the creative workflow.  
- **Key Features** (anticipated):  
  - MIDI file upload or in-browser creation.  
  - Model selection and generation of extended melodies.  
  - Playback, download, or automated piano roll video visualisation of the produced continuation.

---

## Usage

1. **Install Dependencies**  
   - Make sure you have all the required Python packages installed:  
     ```bash
     pip install -r requirements.txt
     ```

2. **Set Up FluidSynth and SoundFont**  
   - Install FluidSynth and ensure it’s available in your system’s PATH.  
   - Place your SoundFont file (`.sf2`) in a known directory and update the script paths accordingly.

3. **Piano Roll Video Generation**  
   - Adjust configuration variables in the script if needed.  
   - Run:  
     ```bash
     python piano_roll_video.py
     ```
   - The resulting `.mp4` will appear in the specified output directory.

4. **User Testing Front End**  
   - Update any database or local configuration.  
   - Run:  
     ```bash
     streamlit run app.py
     ```
   - Follow the link provided in the console to access the application in your web browser.

5. **AI Music Assistant Front End**  
   - When completed, it will similarly be run with Streamlit:  
     ```bash
     streamlit run app2.py
     ```
   - This will allow you to upload or create MIDI files, generate continuations, and download or view your results.

---

## Licence

This project is for academic and research purposes. Please consult the dissertation’s guidelines or the project supervisor for details on usage and sharing.  
