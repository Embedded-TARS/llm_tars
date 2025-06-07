import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.manual_seed(0)

model_path = "microsoft/Phi-4-mini-instruct"

# 모델 및 토크나이저 로딩
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 파이프라인 생성
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# 대화 히스토리 초기화
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."}
]

generation_args = {
    "max_new_tokens": 256,
    "return_full_text": False,
    "temperature": 0.7,
    "do_sample": False,
}

print("💬 대화를 시작하세요. 종료하려면 'exit' 입력")

while True:
    user_input = input("👤 You: ")
    if user_input.strip().lower() in ['exit', 'quit']:
        print("👋 대화를 종료합니다.")
        break

    messages.append({"role": "user", "content": user_input})

    # 모델 호출
    output = pipe(messages, **generation_args)
    reply = output[0]["generated_text"]
    print(f"🤖 AI: {reply.strip()}")

    messages.append({"role": "assistant", "content": reply.strip()})

