import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel  # For LoRA models


def generate_dialogue(prompt, model_path="./models/dialogue_generator_gpt2"):
    # Check for empty prompt
    if not prompt or prompt.strip() == "":
        return "Prompt cannot be empty."
        
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Apply better generation parameters
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=512,  # Longer outputs
        temperature=0.7,  # Good balance of creativity and coherence
        do_sample=True,
        top_p=0.9,
        top_k=40,
        repetition_penalty=1.2,  # Avoid repetitive text
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


# Example usage (only runs when script is executed directly)
if __name__ == "__main__":
    dialogue = generate_dialogue("Generate dialogue for scene 'Forest Encounter':")
    print(dialogue)