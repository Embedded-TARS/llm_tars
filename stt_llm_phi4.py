import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import whisper
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from gtts import gTTS
import os
import json

SAMPLE_RATE = 16000
CHANNELS = 1
OUTPUT_FILE = "recorded_audio.wav"
recording = []

def audio_callback(indata, frames, time_info, status):
    recording.append(indata.copy())

def record_audio():
    global recording
    recording = []
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback):
        input()

    audio_data = np.concatenate(recording, axis=0)
    wav.write(OUTPUT_FILE, SAMPLE_RATE, audio_data)

def transcribe_audio():
    model = whisper.load_model("tiny.en")
    result = model.transcribe(OUTPUT_FILE)
    return result["text"]

torch.manual_seed(0)
model_path = "microsoft/Phi-4-mini-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
'''
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."}
    ]

'''
messages = [
    {
        "role": "system",
        "content": (
            "You are a robotic driving assistant that interprets Korean user commands "
            "into structured JSON control instructions for a rover.\n\n"
            "There are only two types of tasks:\n"
            "1. `navigate`: Go to a specific destination with a speed setting.\n"
            "   - Valid destinations: 'home', 'office', 'airport', 'school'\n"
            "   - Valid speeds: 'fast', 'normal' (default), 'slow'\n\n"
            "2. `manual_command`: Direct movement commands.\n"
            "   - Valid commands (mapped to action field):\n"
            "     - 'stop' → 'stop'\n"
            "     - 'forward' → 'go_forward'\n"
            "     - 'backward' → 'go_backward'\n"
            "     - 'left_turn' → 'turn_left'\n"
            "     - 'right_turn' → 'turn_right'\n"
            "     - 'turn_around' → 'turn_around'\n\n"
            "If the user's command is unclear or doesn't match any category, reply politely asking for clarification.\n\n"
            "You must always respond with:\n"
            "1. A short assistant-style reply in English (e.g., 'Okay, going to school at normal speed.')\n"
            "2. A JSON block enclosed in triple backticks, like this:\n"
            "```\n"
            "{\n"
            '  "task_type": "navigate" | "manual_command" | "unknown",\n'
            '  "action": "navigate_to" | "stop" | "go_forward" | "go_backward" | "turn_left" | "turn_right" | "turn_around" | "",\n'
            '  "parameters": {\n'
            '     "destination": "home" | "office" | "airport" | "school" | null,\n'
            '     "speed": "fast" | "normal" | "slow" | null\n'
            '  }\n'
            "}\n"
            "```\n"
            "Only use the values listed above. If the input is unclear (e.g., 'go anywhere'), then return 'task_type': 'unknown', and ask the user to clarify."
        )
    }
]



generation_args = {
    "max_new_tokens": 128,
    "return_full_text": False,
    #"temperature": 0.7,
    "do_sample": False,
}

print("[Enter] 키를 눌러 녹음을 시작/종료하세요.")

while True:
    user_input = input("[Enter]로 녹음 / 'exit' 입력시 종료: ").strip().lower()
    if user_input == "exit":
        print("대화를 종료합니다.")
        break

    record_audio()
    stt_text = transcribe_audio()
    print(f"You: {stt_text}")

    messages.append({"role": "user", "content": stt_text})
    output = pipe(messages, **generation_args)
    reply = output[0]["generated_text"].strip()
    
    lines = reply.strip().split("\n")
    text_response = lines[0]
    tts = gTTS(text=text_response, lang='en')
    tts.save("response.mp3")
    os.system("mpg123 response.mp3")
    
    print(f"TARS: {reply}")

    messages.append({"role": "assistant", "content": reply})
