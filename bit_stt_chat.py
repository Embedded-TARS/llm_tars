import torch
import torch._dynamo
# Completely disable PyTorch compilation to avoid warnings
torch._dynamo.config.disable = True
from transformers import AutoModelForCausalLM, AutoTokenizer
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import whisper
import time
import warnings
import sys
import os

warnings.filterwarnings("ignore", category=FutureWarning)

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
OUTPUT_FILE = "recorded_audio.wav"

class VoiceChat:
    def __init__(self):
        self.recording = []
        self.model = None
        self.tokenizer = None
        self.whisper_model = None
        self.messages = [
            {"role": "system", "content": "You are a helpful AI assistant. Keep your responses concise and conversational."}
        ]
    
    def load_models(self):
        """Load both BitNet and Whisper models"""
        print("🔄 Loading models...")
        
        # Load Whisper model
        print("📱 Loading Whisper STT model...")
        self.whisper_model = whisper.load_model("tiny.en")
        print("✅ Whisper model loaded!")
        
        # Load BitNet model
        model_id = "microsoft/bitnet-b1.58-2B-4T"
        print("🤖 Loading BitNet tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        print("🤖 Loading BitNet model (this may take a while)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"}
        )
        
        print(f"✅ BitNet model loaded on device: {self.model.device}")
        print(f"📊 Model dtype: {self.model.dtype}")
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio recording"""
        self.recording.append(indata.copy())
    
    def record_audio(self):
        """Record audio from microphone"""
        self.recording = []
        print("🎤 Recording... Press [Enter] to stop.")
        
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=self.audio_callback):
            input()
        
        audio_data = np.concatenate(self.recording, axis=0)
        wav.write(OUTPUT_FILE, SAMPLE_RATE, audio_data)
        print("🎵 Audio recorded!")
    
    def transcribe_audio(self):
        """Convert speech to text using Whisper"""
        if not os.path.exists(OUTPUT_FILE):
            return None
        
        print("🔄 Converting speech to text...")
        result = self.whisper_model.transcribe(OUTPUT_FILE)
        transcribed_text = result["text"].strip()
        
        if transcribed_text:
            print(f"📝 You said: {transcribed_text}")
            return transcribed_text
        else:
            print("❌ No speech detected, please try again.")
            return None
    
    def generate_response(self, user_input, max_tokens=100):
        """Generate response from BitNet model"""
        try:
            # Add user message to conversation
            self.messages.append({"role": "user", "content": user_input})
            
            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                self.messages, tokenize=False, add_generation_prompt=True
            )
            chat_input = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            print("🤖 Generating response...")
            
            # Generate response
            with torch.no_grad():
                chat_outputs = self.model.generate(
                    **chat_input, 
                    max_new_tokens=max_tokens, 
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(
                chat_outputs[0][chat_input['input_ids'].shape[-1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Add response to conversation
            self.messages.append({"role": "assistant", "content": response})
            
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def handle_text_commands(self, text):
        """Handle special text commands"""
        text_lower = text.lower().strip()
        
        if any(word in text_lower for word in ['quit', 'exit', 'bye', 'goodbye']):
            return 'quit'
        elif any(word in text_lower for word in ['clear', 'reset', 'new conversation']):
            return 'clear'
        elif 'help' in text_lower:
            return 'help'
        
        return None
    
    def run(self):
        """Main conversation loop"""
        print("🎙️ Voice Chat with BitNet")
        print("=" * 40)
        
        try:
            self.load_models()
        except Exception as e:
            print(f"❌ Failed to load models: {e}")
            sys.exit(1)
        
        print("\n✅ All models loaded successfully!")
        print("\n💬 Voice Chat Instructions:")
        print("  • Press [Enter] to start recording")
        print("  • Speak your message")
        print("  • Press [Enter] again to stop recording")
        print("  • Say 'quit' or 'goodbye' to exit")
        print("  • Say 'clear' to reset conversation")
        print("  • Say 'help' for commands")
        print("-" * 40)
        
        while True:
            try:
                # Record audio
                input("\n🎤 Press [Enter] to start recording your message...")
                self.record_audio()
                
                # Transcribe audio
                transcribed_text = self.transcribe_audio()
                
                if not transcribed_text:
                    continue
                
                # Handle commands
                command = self.handle_text_commands(transcribed_text)
                
                if command == 'quit':
                    print("👋 Goodbye!")
                    break
                elif command == 'clear':
                    self.messages = [
                        {"role": "system", "content": "You are a helpful AI assistant. Keep your responses concise and conversational."}
                    ]
                    print("🗑️ Conversation history cleared!")
                    continue
                elif command == 'help':
                    print("\n📋 Voice Commands:")
                    print("  • Say 'quit' or 'goodbye' to exit")
                    print("  • Say 'clear' to reset conversation")
                    print("  • Say 'help' for this message")
                    continue
                
                # Generate and display response
                response = self.generate_response(transcribed_text)
                print(f"\n🤖 Assistant: {response}")
                
                # Limit conversation history
                if len(self.messages) > 20:
                    self.messages = self.messages[:1] + self.messages[-19:]
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("🔄 Continuing conversation...")
        
        # Cleanup
        if os.path.exists(OUTPUT_FILE):
            os.remove(OUTPUT_FILE)
            print("🧹 Temporary audio file cleaned up.")

def main():
    chat = VoiceChat()
    chat.run()

if __name__ == "__main__":
    main()
