import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import torch

def main():
    # Define model and parameters - ENHANCED FOR BETTER CAPABILITY
    model_name = "gpt2"  #   gpt2 
    output_dir = "models/dialogue_generator_full"  # Different output directory
    batch_size = 4  # Increased batch size for better gradient estimates
    num_epochs = 5  # More training epochs for better learning
    learning_rate = 2e-5  # Slightly lower learning rate for better convergence
    
    # Set up device - avoid MPS issues by using CPU if needed
    if torch.backends.mps.is_available():
        try:
            # Test MPS with a small tensor operation
            x = torch.zeros(1).to('mps')
            device = 'mps'
            print("Using MPS device")
        except:
            device = 'cpu'
            print("MPS available but encountered issues, falling back to CPU")
    else:
        device = 'cpu'
        print("MPS not available, using CPU")
    
    # Load the data
    print("Loading data...")
    
    # Update paths to match your directory structure
    base_data_dir = "./data/processed"  # Updated path to point to data directory
    
    # First check if the formatted files exist
    if os.path.exists(f"{base_data_dir}/train_formatted.csv"):
        train_df = pd.read_csv(f"{base_data_dir}/train_formatted.csv")
        val_df = pd.read_csv(f"{base_data_dir}/val_formatted.csv")
    else:
        # If not, use the raw1 files
        if os.path.exists(f"{base_data_dir}/raw1_train.csv"):
            train_df = pd.read_csv(f"{base_data_dir}/raw1_train.csv")
            val_df = pd.read_csv(f"{base_data_dir}/raw1_val.csv")
        else:
            # If neither exists, use the raw1_dialogues to create quick train/val sets
            print(f"Creating quick train/val splits from {base_data_dir}/raw1_dialogues.csv")
            
            # Make sure the directory exists before trying to access files
            if not os.path.exists(base_data_dir):
                os.makedirs(base_data_dir, exist_ok=True)
                raise FileNotFoundError(f"Directory {base_data_dir} was created, but no data files exist yet. Please run process_data.py first.")
                
            if not os.path.exists(f"{base_data_dir}/raw1_dialogues.csv"):
                raise FileNotFoundError(f"File {base_data_dir}/raw1_dialogues.csv not found. Please run process_data.py first.")
                
            dialogues_df = pd.read_csv(f"{base_data_dir}/raw1_dialogues.csv")
            
            # Create training examples with basic prompts
            examples = []
            for _, row in dialogues_df.iterrows():
                # Check which column name is actually used - source_file or source
                source_col = 'source_file' if 'source_file' in dialogues_df.columns else 'source'
                if source_col not in dialogues_df.columns:
                    print(f"Warning: Neither 'source_file' nor 'source' column found in dialogues CSV.")
                    print(f"Available columns: {dialogues_df.columns.tolist()}")
                    source = "unknown"
                else:
                    source = row[source_col]
                    if isinstance(source, str) and source.endswith('.json'):
                        source = source.replace('.json', '')
                    
                dialogue = row['dialogue']
                if isinstance(dialogue, str) and dialogue.strip():
                    prompt = f"Generate dialogue for scene '{source}':"
                    example_dict = {
                        'source': source,
                        'prompt': prompt,
                        'completion': dialogue,
                        'full_text': f"{prompt}\n\n{dialogue}"
                    }
                    
                    # Add dialogueID if it exists
                    if 'dialogueID' in dialogues_df.columns:
                        example_dict['dialogueID'] = row['dialogueID']
                        
                    examples.append(example_dict)
            
            # Convert to DataFrame and split
            examples_df = pd.DataFrame(examples)
            
            if examples_df.empty:
                raise ValueError("No valid examples could be created from the dialogues file.")
                
            examples_df = examples_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Use 80% for train, 20% for validation
            split_idx = int(0.8 * len(examples_df))
            train_df = examples_df[:split_idx]
            val_df = examples_df[split_idx:]
            
            # Save the train/val splits for future use
            train_df.to_csv(f"{base_data_dir}/raw1_train.csv", index=False)
            val_df.to_csv(f"{base_data_dir}/raw1_val.csv", index=False)
            print(f"Saved train/val splits to {base_data_dir}")
    
    # Use more data for the full model - up to 100 examples
    if not train_df.empty:
        train_df = train_df.sample(min(100, len(train_df)), random_state=42)
    if not val_df.empty:
        val_df = val_df.sample(min(20, len(val_df)), random_state=42)
    
    print(f"Using {len(train_df)} training examples and {len(val_df)} validation examples")
    
    # Show some examples of the data
    print("\nSample training example:")
    if not train_df.empty:
        if 'full_text' in train_df.columns:
            print(train_df['full_text'].iloc[0][:200] + "...")
        else:
            print("WARNING: 'full_text' column not found in data")
            # Show what columns we do have
            print(f"Available columns: {train_df.columns.tolist()}")
            # Try to adapt to available columns
            if 'prompt' in train_df.columns and 'completion' in train_df.columns:
                train_df['full_text'] = train_df['prompt'] + "\n\n" + train_df['completion']
                val_df['full_text'] = val_df['prompt'] + "\n\n" + val_df['completion']
            else:
                raise ValueError("Data does not have required columns for training")
    else:
        raise ValueError("No training data available")
    
    # Convert to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    # Load tokenizer and model
    print(f"Loading {model_name} model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    # If the tokenizer doesn't have a pad token, set it
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Function to tokenize inputs for causal language modeling - increased context window
    def tokenize_function(examples):
        # Tokenize the text with longer sequence length
        outputs = tokenizer(
            examples["full_text"], 
            padding="max_length", 
            truncation=True, 
            max_length=512,  # Increased from 256 for longer context
            return_tensors=None  # Return Python lists instead of tensors
        )
        
        # The labels are the same as inputs for causal LM
        outputs["labels"] = outputs["input_ids"].copy()
        
        return outputs
    
    # Apply tokenization
    print("Tokenizing datasets...")
    train_tokenized = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
    val_tokenized = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal language modeling, not masked language modeling
    )
    
    # Set up training arguments with device specification - improved settings
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=50,  # Increased from 10 for better training stability
        weight_decay=0.05,  # Increased from 0.01 for better regularization
        logging_dir="./logs",
        logging_steps=1,
        # Remove evaluation_strategy, eval_steps, and load_best_model_at_end
        # evaluation_strategy="steps",
        # eval_steps=100,
        save_steps=50,  # Save more frequently
        save_total_limit=3,  # Keep only the 3 best checkpoints
        learning_rate=learning_rate,
        no_cuda=True if device == 'cpu' else False,  # Don't use CUDA if we're on CPU
        fp16=False,  # Disable mixed precision to avoid MPS issues
        # load_best_model_at_end=True
    )
    
    # Set up trainer with data collator
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    
    # Generate a sample from the trained model with improved generation parameters
    print("\nGenerating a sample from the trained model:")
    sample_prompt = "Generate dialogue for scene 'Test Scene':"
    inputs = tokenizer(sample_prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs, 
        max_length=200,  # Increased from 100 for longer outputs
        temperature=0.8,  # Slightly higher temperature for more creativity
        num_return_sequences=1,
        do_sample=True,
        top_p=0.92,
        no_repeat_ngram_size=2,
        top_k=50
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n" + "="*40)
    print(generated_text)
    print("="*40)

if __name__ == "__main__":
    main()