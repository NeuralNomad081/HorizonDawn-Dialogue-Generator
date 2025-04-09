import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def generate_dialogue(prompt, model_path="models/trained_model", max_length=512):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate text
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.9,
        top_p=0.95,
        do_sample=True,
    )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Generate game dialogue")
    parser.add_argument("--prompt", type=str, default="Generate dialogue for game scene 'Mysterious Cave':", 
                       help="Prompt for dialogue generation")
    parser.add_argument("--model_path", type=str, default="models/trained_model",
                       help="Path to the trained model")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum length of generated text")
    
    args = parser.parse_args()
    
    # Generate dialogue
    print("Generating dialogue...")
    generated = generate_dialogue(args.prompt, args.model_path, args.max_length)
    
    print("\n" + "="*50)
    print(generated)
    print("="*50 + "\n")

if __name__ == "__main__":
    main()