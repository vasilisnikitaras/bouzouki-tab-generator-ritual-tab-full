import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import yt_dlp
import librosa
import soundfile as sf
import os
import re
from fpdf import FPDF
from mido import Message, MidiFile, MidiTrack
import librosa.display
import base64
import pandas as pd

# Page setup
st.set_page_config(page_title="Τελετουργική Ταμπλατούρα", page_icon="🎼")
st.title("🎼 Τελετουργική Ταμπλατούρα για Τετράχορδο Μπουζούκι")
st.markdown("Καλώς ήρθες στην τελετουργική εφαρμογή για μετατροπή νοτών, συχνοτήτων και τραγουδιών σε ταμπλατούρα για τετράχορδο μπουζούκι.")

# String bases and names
string_bases = {'Ντο': 48, 'Φα': 53, 'Λα': 57, 'Ρε': 62}
note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
greek_names = {
    'C': 'Ντο', 'C#': 'Ντο#', 'D': 'Ρε', 'D#': 'Ρε#', 'E': 'Μι',
    'F': 'Φα', 'F#': 'Φα#', 'G': 'Σολ', 'G#': 'Σολ#', 'A': 'Λα',
    'A#': 'Λα#', 'B': 'Σι'
}

# Conversions
def freq_to_midi(freq):
    return int(round(69 + 12 * np.log2(freq / 440.0)))

def midi_to_freq(midi):
    return round(440 * 2 ** ((midi - 69) / 12), 2)

def midi_to_note(midi):
    name = note_names[midi % 12]
    octave = midi // 12 - 1
    greek = greek_names.get(name, name)
    return f"{name}{octave} / {greek} / MIDI:{midi} / {midi_to_freq(midi)}Hz"

def note_to_midi(note):
    match = re.match(r'^([A-G]#?|[A-G]b?)(-?\d+)$', note.strip())
    if not match:
        raise ValueError(f"Μη έγκυρη νότα: {note}")
    name, octave = match.groups()
    return note_names.index(name) + 12 * (int(octave) + 1)

def find_positions(midi):
    return [(s, midi - b) for s, b in string_bases.items() if 0 <= midi - b <= 12]

# Plot fretboard positions
def plot_positions(midi):
    positions = find_positions(midi)
    fig, ax = plt.subplots(figsize=(10, 4))
    strings = list(string_bases.keys())
    ax.set_yticks(range(len(strings)))
    ax.set_yticklabels(strings)
    ax.set_xticks(range(13))
    ax.grid(True)
    for s, f in positions:
        y = strings.index(s)
        ax.plot(f, y, 'ro', markersize=12)
        ax.text(f, y + 0.2, midi_to_note(midi), ha='center')
    st.pyplot(fig)

# Tab generation from notes
def tab_from_notes(note_list):
    tab = []
    for note, dur in note_list:
        try:
            midi = note_to_midi(note)
            pos = find_positions(midi)
            if pos:
                s, f = pos[0]
                tab.append({
                    'Νότα': midi_to_note(midi),
                    'Χορδή': s,
                    'Τάστο': f,
                    'Διάρκεια': dur
                })
            else:
                tab.append({
                    'Νότα': midi_to_note(midi),
                    'Χορδή': '—',
                    'Τάστο': '—',
                    'Διάρκεια': dur
                })
        except:
            continue
    return tab

# PDF export
def generate_pdf(tab):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="🎼 Τελετουργική Ταμπλατούρα", ln=True, align='C')
    for t in tab:
        line = f"{t['Νότα']} → Χορδή: {t['Χορδή']}, Τάστο: {t['Τάστο']}, Διάρκεια: {t['Διάρκεια']}"
        pdf.cell(200, 10, txt=line, ln=True)
    pdf.output("tab.pdf")
    return "tab.pdf"

# MIDI export
def export_midi(tab, filename="output.mid"):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    for t in tab:
        try:
            midi_num_part = t['Νότα'].split('/')[0]  # e.g., "C4 "
            # extract letter+octave safely
            match = re.match(r'^([A-G]#?)(-?\d)', midi_num_part.strip())
            if not match:
                continue
            note_str = "".join(match.groups())
            midi = note_to_midi(note_str)
            duration = int(float(t['Διάρκεια']) * 480)
            track.append(Message('note_on', note=midi, velocity=64, time=0))
            track.append(Message('note_off', note=midi, velocity=64, time=duration))
        except:
            continue
    mid.save(filename)
    return filename

# Spectrum plot
def plot_spectrum(file_path):
    y, sr = librosa.load(file_path)
    D = np.abs(librosa.stft(y))**2
    S = librosa.feature.melspectrogram(S=D, sr=sr)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        librosa.power_to_db(S, ref=np.max),
        sr=sr, x_axis='time', y_axis='mel'
    )
    ax.set_title("📈 Φασματική Ανάλυση")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig)

# YouTube audio download
def download_youtube_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192'
        }],
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return 'audio.wav'

# Note extraction with timing
def extract_notes_with_timing(file_path):
    y, sr = librosa.load(file_path)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    notes = []
    times = librosa.frames_to_time(np.arange(pitches.shape[1]), sr=sr)

    for i in range(pitches.shape[1] - 1):
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        if pitch > 0:
            note = librosa.hz_to_note(pitch)
            start_time = round(times[i], 2)
            end_time = round(times[i + 1], 2)
            duration = round(end_time - start_time, 2)
            notes.append((note, start_time, duration))

    return notes[:50]

# Clipboard helper for Suno prompt
def clipboard_button(text, label="📋 Αντιγραφή Prompt"):
    b64 = base64.b64encode(text.encode()).decode()
    button_html = f"""
    <button onclick="navigator.clipboard.writeText(atob('{b64}'))">{label}</button>
    """
    st.markdown(button_html, unsafe_allow_html=True)

# App state
tab = []
input_type = st.radio("📥 Επιλέξτε είδος εισόδου:",
                      ["Νότα", "Συχνότητα", "Αρχείο Ήχου", "YouTube", "Αρχείο TXT"])

# Note input block
try:
    if input_type == "Νότα":
        note_in = st.text_input("🎵 Εισάγετε νότα (π.χ. G4):")
        dur = st.number_input("⏱️ Διάρκεια (s):", min_value=0.1, value=1.0)
        if note_in:
            st.write(f"Νότα: {note_in} — Διάρκεια: {dur}s")
            try:
                midi = note_to_midi(note_in)
                plot_positions(midi)
            except Exception as e:
                st.error(f"Σφάλμα νότας: {e}")
            tab = tab_from_notes([(note_in, dur)])
except Exception as e:
    st.error(f"⚠️ Σφάλμα Νότας: {e}")

# Frequency input block
try:
    if input_type == "Συχνότητα":
        freq_in = st.number_input("📡 Εισάγετε συχνότητα (Hz):", min_value=1.0, value=440.0)
        dur = st.number_input("⏱️ Διάρκεια (s):", min_value=0.1, value=1.0, key="freq_dur")
        if freq_in:
            midi = freq_to_midi(freq_in)
            st.write(f"Συχνότητα: {freq_in}Hz → {midi_to_note(midi)}")
            plot_positions(midi)
            # Derive a note name for tab_from_notes
            name = note_names[midi % 12] + str(midi // 12 - 1)
            tab = tab_from_notes([(name, dur)])
except Exception as e:
    st.error(f"⚠️ Σφάλμα Συχνότητας: {e}")

# Audio file block
try:
    if input_type == "Αρχείο Ήχου":
        audio_file = st.file_uploader("🎙️ Ανεβάστε αρχείο .wav", type=["wav"])
        if audio_file:
            temp_path = "uploaded.wav"
            with open(temp_path, "wb") as f:
                f.write(audio_file.read())
            st.audio(temp_path, format="audio/wav")
            plot_spectrum(temp_path)
            notes = extract_notes_with_timing(temp_path)
            if notes:
                st.markdown("🎵 **Αναγνωρισμένες Νότες από Ήχο:**")
                for note, ts, dur in notes:
                    st.write(f"• {note} @ {ts}s → διάρκεια: {dur}s")
                df_notes = pd.DataFrame(notes, columns=["Νότα", "Χρόνος (s)", "Διάρκεια (s)"])
                st.dataframe(df_notes)
            tab = tab_from_notes([(n, d) for n, ts, d in notes])
except Exception as e:
    st.error(f"⚠️ Σφάλμα Αρχείου Ήχου: {e}")

# YouTube block
try:
    if input_type == "YouTube":
        url = st.text_input("📺 Εισάγετε σύνδεσμο YouTube:")
        if url:
            st.write("🔄 Λήψη και ανάλυση...")
            audio_path = download_youtube_audio(url)
            st.audio(audio_path, format='audio/wav')
            with open(audio_path, "rb") as f:
                st.download_button("📥 Κατέβασε το αρχείο ήχου (audio.wav)", f, file_name="audio.wav")
            plot_spectrum(audio_path)
            notes = extract_notes_with_timing(audio_path)
            if notes:
                st.markdown("🎵 **Αναγνωρισμένες Νότες από YouTube:**")
                for note, ts, dur in notes:
                    st.write(f"• {note} @ {ts}s → διάρκεια: {dur}s")
                df_notes = pd.DataFrame(notes, columns=["Νότα", "Χρόνος (s)", "Διάρκεια (s)"])
                st.dataframe(df_notes)
            tab = tab_from_notes([(note, dur) for note, ts, dur in notes])
except Exception as e:
    st.error(f"⚠️ Σφάλμα YouTube: {e}")

# TXT block
try:
    if input_type == "Αρχείο TXT":
        txt_file = st.file_uploader("📄 Ανεβάστε αρχείο .txt με νότες και διάρκειες (π.χ. G4 1.0)", type=["txt"])
        if txt_file:
            content = txt_file.read().decode("utf-8")
            lines = content.strip().split("\n")
            note_list = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 2:
                    note, dur = parts
                    try:
                        note_list.append((note, float(dur)))
                    except:
                        continue
            tab = tab_from_notes(note_list)
            if note_list:
                st.markdown("🎼 **Νότες από TXT:**")
                for note, dur in note_list:
                    st.write(f"• {note} → Διάρκεια: {dur}s")
                df_txt = pd.DataFrame(note_list, columns=["Νότα", "Διάρκεια (s)"])
                st.dataframe(df_txt)
except Exception as e:
    st.error(f"⚠️ Σφάλμα TXT: {e}")

# Suno prompt
try:
    st.markdown("🌞 Δημιουργία Prompt για Suno:")
    suno_prompt = st.text_area("🎤 Περιγραφή τραγουδιού για Suno:", key="suno_prompt")
    if suno_prompt:
        st.success(f"🌞 Prompt έτοιμο: {suno_prompt}")
        clipboard_button(suno_prompt)
        st.markdown("[🎵 Άνοιξε το Suno Studio](https://suno.com/me)")
except Exception as e:
    st.error(f"⚠️ Σφάλμα Suno: {e}")

# PDF export
try:
    if tab:
        if st.button("📄 Εξαγωγή PDF Ταμπλατούρας"):
            pdf_path = generate_pdf(tab)
            with open(pdf_path, "rb") as f:
                st.download_button("📥 Κατέβασε PDF", f, file_name="tab.pdf")
except Exception as e:
    st.error(f"⚠️ Σφάλμα PDF: {e}")

# MIDI export
try:
    if tab:
        if st.button("🎼 Εξαγωγή MIDI"):
            midi_path = export_midi(tab)
            with open(midi_path, "rb") as f:
                st.download_button("📥 Κατέβασε MIDI", f, file_name="output.mid")
except Exception as e:
    st.error(f"⚠️ Σφάλμα MIDI: {e}")

# Tab display
if tab:
    st.markdown("🎼 **Ταμπλατούρα:**")
    for t in tab:
        st.write(f"{t['Νότα']} → Χορδή: {t['Χορδή']}, Τάστο: {t['Τάστο']}, Διάρκεια: {t['Διάρκεια']}")

