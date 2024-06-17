#!/usr/bin/env python3
"""
Created by: Evan Beaudoin
Created on: June 2024
This is the final Song Maker 3000
"""

from flask import Flask, render_template, request, jsonify
import threading
import time
import numpy as np
import sounddevice as sd
import copy
import scipy.fftpack  
import os 

# pylint: disable=C0103
app = Flask(__name__)

# General settings constants for audio processing
SAMPLE_FREQUENCY_HZ = 48000  # Sample frequency in Hz
DFT_WINDOW_SIZE_SAMPLES = 48000  # Window size of the DFT in samples
DFT_WINDOW_STEP_SIZE_SAMPLES = 12000  # Step size of the window
MAX_NUM_HARMONIC_PRODUCTS = 5  # Max number of harmonic product spectrums
SIGNAL_POWER_THRESHOLD = 1e-6  # Recording is activated if the signal power exceeds this threshold
STANDARD_CONCERT_PITCH_HZ = 440  # Defining A4 pitch frequency
WHITE_NOISE_THRESHOLD = 0.2  # Threshold for cutting off low magnitude frequencies

WINDOW_LENGTH_SECONDS = DFT_WINDOW_SIZE_SAMPLES / SAMPLE_FREQUENCY_HZ  # Length of the window in seconds
SAMPLE_PERIOD_SECONDS = 1 / SAMPLE_FREQUENCY_HZ  # Period between two samples in seconds
FREQUENCY_STEP_WIDTH_HZ = SAMPLE_FREQUENCY_HZ / DFT_WINDOW_SIZE_SAMPLES  # Frequency step width of the interpolated DFT
OCTAVE_BANDS_HZ = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]

ALL_NOTES_LIST = ["A","A#","B","C","C#","D","D#","E","F","F#","G","G#"]
NOTE_TO_TABLATURE = {
    'E2': (6, 0), 'F2': (6, 1), 'F#2': (6, 2), 'G2': (6, 3), 'G#2': (6, 4), 'A2': (6, 5), 'A#2': (6, 6), 'B2': (6, 7), 'C3': (6, 8), 'C#3': (6, 9), 'D3': (6, 10), 'D#3': (6, 11), 'E3': (6, 12),
    'A2': (5, 0), 'A#2': (5, 1), 'B2': (5, 2), 'C3': (5, 3), 'C#3': (5, 4), 'D3': (5, 5), 'D#3': (5, 6), 'E3': (5, 7), 'F3': (5, 8), 'F#3': (5, 9), 'G3': (5, 10), 'G#3': (5, 11), 'A3': (5, 12),
    'D3': (4, 0), 'D#3': (4, 1), 'E3': (4, 2), 'F3': (4, 3), 'F#3': (4, 4), 'G3': (4, 5), 'G#3': (4, 6), 'A3': (4, 7), 'A#3': (4, 8), 'B3': (4, 9), 'C4': (4, 10), 'C#4': (4, 11), 'D4': (4, 12),
    'G3': (3, 0), 'G#3': (3, 1), 'A3': (3, 2), 'A#3': (3, 3), 'B3': (3, 4), 'C4': (3, 5), 'C#4': (3, 6), 'D4': (3, 7), 'D#4': (3, 8), 'E4': (3, 9), 'F4': (3, 10), 'F#4': (3, 11), 'G4': (3, 12),
    'B3': (2, 0), 'C4': (2, 1), 'C#4': (2, 2), 'D4': (2, 3), 'D#4': (2, 4), 'E4': (2, 5), 'F4': (2, 6), 'F#4': (2, 7), 'G4': (2, 8), 'G#4': (2, 9), 'A4': (2, 10), 'A#4': (2, 11), 'B4': (2, 12),
    'E4': (1, 0), 'F4': (1, 1), 'F#4': (1, 2), 'G4': (1, 3), 'G#4': (1, 4), 'A4': (1, 5), 'A#4': (1, 6), 'B4': (1, 7), 'C5': (1, 8), 'C#5': (1, 9), 'D5': (1, 10), 'D#5': (1, 11), 'E5': (1, 12),
}

def find_closest_note_to_pitch(pitch_frequency):
    """
    Find the closest musical note to the given pitch.

    Args:
    pitch_frequency (float): The frequency of the pitch.

    Returns:
    tuple: Closest note as a string and its pitch as a float.
    """
    note_index = int(np.round(np.log2(pitch_frequency / STANDARD_CONCERT_PITCH_HZ) * 12))
    closest_note = ALL_NOTES_LIST[note_index % 12] + str(4 + (note_index + 9) // 12)
    closest_pitch = STANDARD_CONCERT_PITCH_HZ * 2**(note_index / 12)
    return closest_note, closest_pitch

HANNING_WINDOW = np.hanning(DFT_WINDOW_SIZE_SAMPLES)

# Global variable to store detected notes and tablature
detected_notes_list = []
is_recording_active = False
song_maker_thread_instance = None

def convert_note_to_tab(note):
    """
    Convert a musical note to its corresponding guitar tablature position.

    Args:
    note (str): The musical note.

    Returns:
    tuple: Corresponding string and fret, or None if not found.
    """
    return NOTE_TO_TABLATURE.get(note, None)

def audio_callback(indata, frames, time, status):
    """
    Callback function for real-time audio processing. Analyzes audio data to detect notes.

    Args:
    indata (numpy array): Input audio data.
    frames (int): Number of frames in the audio data.
    time (CData): Time information.
    status (CallbackFlags): Status information.
    """

    if not hasattr(audio_callback, "window_samples"):
        audio_callback.window_samples = [0 for _ in range(DFT_WINDOW_SIZE_SAMPLES)]
    if not hasattr(audio_callback, "note_buffer"):
        audio_callback.note_buffer = ["1", "2"]

    if status:
        print(status)
        return
    if any(indata):
        audio_callback.window_samples = np.concatenate((audio_callback.window_samples, indata[:, 0]))  # append new samples
        audio_callback.window_samples = audio_callback.window_samples[len(indata[:, 0]):]  # remove old samples

        signal_power = (np.linalg.norm(audio_callback.window_samples, ord=2)**2) / len(audio_callback.window_samples)
        if signal_power < SIGNAL_POWER_THRESHOLD:
            return

        hann_windowed_samples = audio_callback.window_samples * HANNING_WINDOW
        magnitude_spectrum = abs(scipy.fftpack.fft(hann_windowed_samples)[:len(hann_windowed_samples)//2])

        for i in range(int(62 / FREQUENCY_STEP_WIDTH_HZ)):
            magnitude_spectrum[i] = 0

        for j in range(len(OCTAVE_BANDS_HZ) - 1):
            index_start = int(OCTAVE_BANDS_HZ[j] / FREQUENCY_STEP_WIDTH_HZ)
            index_end = int(OCTAVE_BANDS_HZ[j + 1] / FREQUENCY_STEP_WIDTH_HZ)
            index_end = index_end if len(magnitude_spectrum) > index_end else len(magnitude_spectrum)
            average_energy_per_frequency = (np.linalg.norm(magnitude_spectrum[index_start:index_end], ord=2)**2) / (index_end - index_start)
            average_energy_per_frequency = average_energy_per_frequency**0.5
            for i in range(index_start, index_end):
                magnitude_spectrum[i] = magnitude_spectrum[i] if magnitude_spectrum[i] > WHITE_NOISE_THRESHOLD * average_energy_per_frequency else 0

                magnitude_spectrum_interpolated = np.interp(np.arange(0, len(magnitude_spectrum), 1 / MAX_NUM_HARMONIC_PRODUCTS), np.arange(0, len(magnitude_spectrum)), magnitude_spectrum)
        magnitude_spectrum_interpolated = magnitude_spectrum_interpolated / np.linalg.norm(magnitude_spectrum_interpolated, ord=2)

        harmonic_product_spectrum = copy.deepcopy(magnitude_spectrum_interpolated)

        for harmonic in range(MAX_NUM_HARMONIC_PRODUCTS):
            tmp_harmonic_product_spectrum = np.multiply(harmonic_product_spectrum[:int(np.ceil(len(magnitude_spectrum_interpolated) / (harmonic + 1)))],
                                                          magnitude_spectrum_interpolated[::(harmonic + 1)])
            if not any(tmp_harmonic_product_spectrum):
                break
            harmonic_product_spectrum = tmp_harmonic_product_spectrum

        max_index = np.argmax(harmonic_product_spectrum)
        max_frequency = max_index * (SAMPLE_FREQUENCY_HZ / DFT_WINDOW_SIZE_SAMPLES) / MAX_NUM_HARMONIC_PRODUCTS

        closest_note, closest_pitch = find_closest_note_to_pitch(max_frequency)
        max_frequency = round(max_frequency, 1)
        closest_pitch = round(closest_pitch, 1)

        audio_callback.note_buffer.insert(0, closest_note)
        audio_callback.note_buffer.pop()

        if audio_callback.note_buffer.count(audio_callback.note_buffer[0]) == len(audio_callback.note_buffer):
            tablature = convert_note_to_tab(closest_note)
            if tablature:
                detected_notes_list.append((closest_note, tablature))
        else:
            return

    else:
        print('No input')

def song_maker_function():
    """
    Function to handle real-time audio recording and note detection.
    Runs in a separate thread.
    """

    global is_recording_active
    try:
        with sd.InputStream(channels=1, callback=audio_callback, blocksize=DFT_WINDOW_STEP_SIZE_SAMPLES, samplerate=SAMPLE_FREQUENCY_HZ):
            while is_recording_active:
                time.sleep(0.5)
    except Exception as e:
        print(str(e))

@app.route('/')
def serve_index_page():
    """
    Serve the main page of the application.

    Returns:
    Rendered HTML template for the index page.
    """
    service_name = os.environ.get('K_SERVICE', 'Unknown service')
    revision_id = os.environ.get('K_REVISION', 'Unknown revision')

    return render_template('index.html', Service=service_name, Revision=revision_id)

@app.route('/start', methods=['POST'])
def start_song_maker_function():
    """
    Start the song maker recording process.

    Returns:
    JSON response indicating the recording status.
    """
    global is_recording_active, song_maker_thread_instance, detected_notes_list
    detected_notes_list = []
    if not is_recording_active:
        is_recording_active = True
        song_maker_thread_instance = threading.Thread(target=song_maker_function)
        song_maker_thread_instance.start()
    return jsonify({"status": "Recording started"})

@app.route('/stop', methods=['POST'])
def stop_song_maker_function():
    """
    Stop the song maker recording process and return detected notes.

    Returns:
    JSON response with the recording status and detected notes.
    """
    global is_recording_active
    is_recording_active = False
    if song_maker_thread_instance:
        song_maker_thread_instance.join()
    notes_output = [{'note': note, 'string': tab[0], 'fret': tab[1]} for note, tab in detected_notes_list]
    return jsonify({"status": "Stopped listening", "notes": notes_output})


if __name__ == '__main__':
    server_port_number = os.environ.get('PORT', '8080')
    app.run(debug=False, port=server_port_number, host='0.0.0.0')

