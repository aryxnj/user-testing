import streamlit as st
import streamlit.components.v1 as components

st.title("Piano Roll with MIDI Export and Audio Playback")

# Embed the interactive Tone.js Piano Roll
components.html("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/tone/14.7.77/Tone.min.js"></script>
    <style>
        #piano-roll {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 500px;
        }
        button {
            margin: 10px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div id="piano-roll">
        <h2>Piano Roll</h2>
        <button onclick="startPlayback()">Play Composition</button>
        <button onclick="exportMIDI()">Export as MIDI</button>
    </div>
    <script>
        const synth = new Tone.PolySynth().toDestination();
        const notes = [
            { time: "0:0:0", note: "C4", duration: "8n" },
            { time: "0:1:0", note: "E4", duration: "8n" },
            { time: "0:2:0", note: "G4", duration: "8n" },
            { time: "0:3:0", note: "B4", duration: "8n" },
        ];

        const part = new Tone.Part((time, value) => {
            synth.triggerAttackRelease(value.note, value.duration, time);
        }, notes).start(0);

        part.loop = true;
        part.loopEnd = "1m";

        function startPlayback() {
            Tone.Transport.start();
        }

        function stopPlayback() {
            Tone.Transport.stop();
        }

        async function exportMIDI() {
            // Convert notes to MIDI
            const midi = new Tone.Midi();
            notes.forEach((n) => {
                midi.addNote({
                    time: n.time,
                    name: n.note,
                    duration: n.duration,
                });
            });

            const blob = new Blob([midi.toArray()], { type: "audio/midi" });
            const url = URL.createObjectURL(blob);
            
            const link = document.createElement("a");
            link.href = url;
            link.download = "composition.mid";
            link.click();
            URL.revokeObjectURL(url);
        }
    </script>
</body>
</html>
""", height=600)
