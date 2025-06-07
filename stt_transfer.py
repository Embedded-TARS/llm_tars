import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import whisper
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

SAMPLE_RATE = 16000
CHANNELS = 1
OUTPUT_FILE = "recorded_audio.wav"
recording = []

def audio_callback(indata, frames, time_info, status):
    recording.append(indata.copy())

def record_audio():
    global recording
    recording = []
    #print("Recording started.")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback):
        input()

    audio_data = np.concatenate(recording, axis=0)
    wav.write(OUTPUT_FILE, SAMPLE_RATE, audio_data)
    #print(f"Recorded file saved: {OUTPUT_FILE}")

def transcribe_audio():
    #print("Whisper model loading...")
    model = whisper.load_model("tiny.en")
    #print("Transfering to text..")
    result = model.transcribe(OUTPUT_FILE)
    #print("Output:")
    print(result["text"])

def main():
    try:
        while True:
            input("Press [Enter] to start recording.")
            record_audio()
            transcribe_audio()
    except KeyboardInterrupt:
        print("\nExit program.")

if __name__ == "__main__":
    main()

