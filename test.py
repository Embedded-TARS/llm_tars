import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import whisper
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")

# Suppress any potential warnings about torch.load
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# -- optional fallback if somebody calls torch.compile explicitly --
def _identity_compile(fn=None, **kwargs):
    if fn is None:
        return lambda f: f
    return fn
torch.compile = _identity_compile

# Setup for audio parameters
SAMPLE_RATE = 16000
CHANNELS = 1
OUTPUT_FILE = "recorded_audio.wav"
recording = []

# Function to handle audio recording callback
def audio_callback(indata, frames, time_info, status):
    global recording
    if status:
        print("Audio recording status:", status)
    recording.append(indata.copy())

# Function to record audio
def record_audio():
    global recording
    recording = []
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback):
            input("🎤 [Enter] to start/stop recording. Type 'exit' to quit: ")
    except Exception as e:
        print(f"Error while recording: {e}")
        return None
    audio_data = np.concatenate(recording, axis=0)
    try:
        wav.write(OUTPUT_FILE, SAMPLE_RATE, audio_data)
        print("Audio saved to", OUTPUT_FILE)
    except Exception as e:
        print(f"Error saving audio: {e}")
        return None
    return OUTPUT_FILE

# Function to transcribe audio using Whisper
def transcribe_audio():
    try:
        model = whisper.load_model("tiny.en")
        result = model.transcribe(OUTPUT_FILE)
        return result["text"]
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

# Load the BitNet model and tokenizer
model_id = "microsoft/bitnet-b1.58-2B-4T"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
    # device_map={"": "cuda:0"}  # Force everything to GPU, no CPU fallback
)

# System messages for the AI
messages = [
    {
        "role": "system",
        "content": (
            "You are a robotic driving assistant that interprets Korean user commands "
            "into structured JSON control instructions for a rover.\n\n"
            "There are only two types of tasks:\n"
            "1. `navigate`: Go to a specific destination with a speed setting.\n"
            "   - Valid destinations: '집' (home), '회사' (office), '공항' (airport), '학교' (school)\n"
            "   - Valid speeds: '빠름' (fast), '보통'(default, normal), '느림' (slow)\n\n"
            "2. `manual_command`: Direct movement commands.\n"
            "   - Valid commands: '멈추기' (stop), '앞으로 가기' (go forward), '뒤로 가기' (go backward),\n"
            "     '왼쪽 회전' (turn left), '오른쪽 회전' (turn right), '뒤돌기' (turn around)\n\n"
            "If the user's command is unclear or doesn't match any category, reply politely asking for clarification.\n\n"
            "Always respond with:\n"
            "1. A short assistant-style reply in English.\n"
            "2. A JSON output in this format:\n"
            "```\n"
            "{\n"
            '  "task_type": "navigate" | "manual_command" | "unknown",\n'
            '  "action": "navigate_to" | "stop" | "go_forward" | "go_backward" | "turn_left" | "turn_right" | "turn_around" | "",\n'
            '  "parameters": {\n'
            '     "destination": "home" | "office" | "airport" | "school" | null,\n'
            '     "speed": "fast" | "normal" | "slow" | null\n'
            '  }\n'
            "}\n"
            "Only use the above values. If the user says something unclear like '거기로 가줘', then ask for clarification and set task_type to 'unknown'."
        )
    }
]

print("🎤 [Enter] 키를 눌러 녹음을 시작/종료하세요. 'exit' 입력 시 종료합니다.")

while True:
    user_input = input("🔘 [Enter]로 녹음 / 'exit' 입력시 종료: ").strip().lower()
    if user_input == "exit":
        print("👋 대화를 종료합니다.")
        break

    # Record audio
    audio_file = record_audio()
    if not audio_file:
        continue

    # Transcribe audio to text
    stt_text = transcribe_audio()
    if not stt_text:
        print("🤖 AI: Could not transcribe audio. Please try again.")
        continue

    print(f"👤 You (STT): {stt_text}")

    # Add user input to messages
    messages.append({"role": "user", "content": stt_text})

    # Tokenize input and generate response
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        chat_input = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate response with max tokens to avoid too long responses
        output = model.generate(**chat_input, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output[0][chat_input['input_ids'].shape[-1]:], skip_special_tokens=True)

        # Truncate response if it's too long
        if len(response) > 200:  # Adjust this limit as needed
            response = response[:200] + "..."

        # Clean up output if necessary (e.g., removing code block markers)
        if response.strip().startswith("```") and response.strip().endswith("```"):
            response = response.strip()[3:-3].strip()

        print(f"🤖 AI: {response.strip()}")
        messages.append({"role": "assistant", "content": response.strip()})
    except Exception as e:
        print(f"Error during model generation: {e}")
