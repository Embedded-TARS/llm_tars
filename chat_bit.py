import torch
import torch._dynamo
# Completely disable PyTorch compilation to avoid warnings
torch._dynamo.config.disable = True
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

def load_model():
    """Load the BitNet model and tokenizer"""
    model_id = "microsoft/bitnet-b1.58-2B-4T"
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print("Loading model (this may take a while on first run)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"}  # Force everything to GPU, no CPU fallback
    )
    
    print(f"Model loaded on device: {model.device}")
    print(f"Model dtype: {model.dtype}")
    return model, tokenizer

def generate_response(model, tokenizer, messages, max_tokens=100):
    """Generate a response from the model"""
    try:
        # Apply chat template
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        chat_input = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate response
        with torch.no_grad():  # Save memory
            chat_outputs = model.generate(
                **chat_input, 
                max_new_tokens=max_tokens, 
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        response = tokenizer.decode(
            chat_outputs[0][chat_input['input_ids'].shape[-1]:], 
            skip_special_tokens=True
        )
        return response.strip()
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

def main():
    print("🤖 BitNet Terminal Chat")
    print("=" * 40)
    
    # Load model
    try:
        model, tokenizer = load_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Initialize conversation history
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Keep your responses concise and helpful."}
    ]
    
    print("\n✅ Model loaded successfully!")
    print("💬 Start chatting! (Type 'quit', 'exit', or 'bye' to end)")
    print("📝 Type 'clear' to clear conversation history")
    print("⚙️  Type 'help' for commands")
    print("-" * 40)
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            elif user_input.lower() == 'clear':
                messages = [
                    {"role": "system", "content": "You are a helpful AI assistant. Keep your responses concise and helpful."}
                ]
                print("🗑️  Conversation history cleared!")
                continue
            
            elif user_input.lower() == 'help':
                print("\n📋 Available commands:")
                print("  • quit/exit/bye - End the conversation")
                print("  • clear - Clear conversation history")
                print("  • help - Show this help message")
                continue
            
            elif not user_input:
                print("❌ Please enter a message!")
                continue
            
            # Add user message to conversation
            messages.append({"role": "user", "content": user_input})
            
            # Generate response
            print("🤖 Assistant: ", end="", flush=True)
            response = generate_response(model, tokenizer, messages)
            print(response)
            
            # Add assistant response to conversation
            messages.append({"role": "assistant", "content": response})
            
            # Optional: Limit conversation history to prevent context overflow
            if len(messages) > 20:  # Keep last 20 messages (including system)
                messages = messages[:1] + messages[-19:]  # Keep system message + last 19
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("🔄 Continuing conversation...")

if __name__ == "__main__":
    main()
