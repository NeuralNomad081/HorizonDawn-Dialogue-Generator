import pandas as pd
import os

def prepare_training_data(input_file="game-content-generator/data/processed/raw1_dialogues.csv", output_dir="game-data-generator/data/processed"):
    """Convert dialogues CSV into proper training format"""
    # Read the dialogues CSV
    df = pd.read_csv(input_file)
    
    # Create training examples with better prompts
    training_data = []
    
    for _, row in df.iterrows():
        source_file = row['source_file'].replace('.json', '')
        dialogue_id = row['dialogueID']
        dialogue = row['dialogue']
        
        if not isinstance(dialogue, str) or dialogue.strip() == "":
            continue
        
        # Extract the title from the first line if it exists
        title = ""
        if "DATAPOINT:" in dialogue:
            first_line = dialogue.split('\n')[0]
            if "DATAPOINT:" in first_line:
                title = first_line.split("DATAPOINT:")[1].strip()
        
        # Create a richer prompt with more context
        if title:
            prompt = f"Generate dialogue for a game scene titled '{title}'. Create an exchange between characters or narration for datapoint '{dialogue_id}':"
        else:
            prompt = f"Generate dialogue for game scene '{dialogue_id}'. Create character interactions or narration for this datapoint:"
        
        training_data.append({
            'source_file': source_file,
            'dialogueID': dialogue_id,
            'prompt': prompt,
            'completion': dialogue,
            'full_text': f"{prompt}\n\n{dialogue}"
        })
    
    # Convert to DataFrame and save
    training_df = pd.DataFrame(training_data)
    
    # Save the formatted training data
    output_path = os.path.join(output_dir, "training_formatted.csv")
    training_df.to_csv(output_path, index=False)
    
    # Create train/val/test splits (80/10/10)
    total = len(training_df)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    
    # Shuffle the data
    training_df = training_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split the data
    train_df = training_df.iloc[:train_size]
    val_df = training_df.iloc[train_size:train_size + val_size]
    test_df = training_df.iloc[train_size + val_size:]
    
    # Save splits
    train_df.to_csv(os.path.join(output_dir, "train_formatted.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_formatted.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_formatted.csv"), index=False)
    
    print(f"Created {total} training examples")
    print(f"Split into {len(train_df)} train, {len(val_df)} val, {len(test_df)} test examples")
    
    return training_df

if __name__ == "__main__":
    prepare_training_data()