import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import whisper
import torch
import os
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 경고 메시지 및 환경 설정
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # 토크나이저 병렬 처리 경고 해결
warnings.filterwarnings("ignore", category=UserWarning)  # 일부 경고 무시

# MPS 사용 시 BFloat16 문제를 피하기 위해 CPU 강제 사용 (선택사항)
# torch.backends.mps.is_available = lambda: False

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

print("🔄 모델을 로딩 중입니다...")

# Apple Silicon 호환성을 위한 모델 로딩
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,  # bfloat16 대신 float16 사용
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print("✅ 모델이 MPS/GPU에서 로딩되었습니다.")
except Exception as e:
    print(f"⚠️  MPS/GPU 로딩 실패, CPU로 fallback: {e}")
    # CPU로 fallback
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cpu",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print("✅ 모델이 CPU에서 로딩되었습니다.")

tokenizer = AutoTokenizer.from_pretrained(model_path)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

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
            "```\n"
            "Only use the above values. If the user says something unclear like '거기로 가줘', then ask for clarification and set task_type to 'unknown'."
        )
    }
]

generation_args = {
    "max_new_tokens": 128,
    "return_full_text": False,
    "do_sample": False,
}

print("🎤 [Enter] 키를 눌러 녹음을 시작/종료하세요. 'exit'를 입력하면 종료됩니다.")

while True:
    user_input = input("🔘 [Enter]로 녹음 / 'exit' 입력시 종료: ").strip().lower()
    
    if user_input == "exit":
        print("👋 대화를 종료합니다.")
        break
    
    print("🎤 녹음 시작... [Enter]를 다시 누르면 종료")
    record_audio()
    print("🔄 음성을 텍스트로 변환 중...")
    
    stt_text = transcribe_audio()
    print(f"👤 You (STT): {stt_text}")
    
    messages.append({"role": "user", "content": stt_text})
    
    print("🤖 AI 응답 생성 중...")
    output = pipe(messages, **generation_args)
    reply = output[0]["generated_text"].strip()
    
    print(f"🤖 AI: {reply}")
    messages.append({"role": "assistant", "content": reply})
