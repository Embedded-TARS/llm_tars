import torch
import torch._dynamo

# Completely disable PyTorch compilation to avoid warnings
torch._dynamo.config.disable = True

from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "microsoft/bitnet-b1.58-2B-4T"

print("Loading tokenizer...")
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Loading model (this may take a while on first run)...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map={"": "cuda:0"}  # Force everything to GPU, no CPU fallback
)

print(f"Model loaded on device: {model.device}")
print(f"Model dtype: {model.dtype}")

print("Preparing chat input...")
# Apply the chat template
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "How are you?"},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
chat_input = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Generating response...")
# Generate response - use sampling to avoid configuration conflicts
chat_outputs = model.generate(
    **chat_input, 
    max_new_tokens=50, 
    do_sample=True,
    temperature=0.1,   # Very low temperature for deterministic-like output
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id
)
response = tokenizer.decode(chat_outputs[0][chat_input['input_ids'].shape[-1]:], skip_special_tokens=True)
print("\nAssistant Response:", response)
