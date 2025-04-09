import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

def process_json_to_csv(raw_dir="data/raw", processed_dir="data/processed"):
    """
    Process JSON files in raw_dir to CSV format for training dialogue generation models.
    
    The script creates:
    - raw1_dialogues.csv: All dialogues from all sources
    - raw1_train.csv: Training split (80% of data)
    - raw1_val.csv: Validation split (20% of data)
    - train_formatted.csv, val_formatted.csv: Same splits but with improved formatting
    """
    print(f"Processing JSON files from {raw_dir} to {processed_dir}")
    
    # Ensure the processed directory exists
    os.makedirs(processed_dir, exist_ok=True)
    
    # Find all JSON files
    json_files = [f for f in os.listdir(raw_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in {raw_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Collect all dialogues
    all_dialogues = []
    
    for file in json_files:
        source_file = file
        file_path = os.path.join(raw_dir, file)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract dialogues based on the structure of your JSON files
            if isinstance(data, dict) and 'dialogues' in data:
                # Format: {"dialogues": [...]}
                dialogues = data['dialogues']
                processed_dialogues = process_standard_dialogues(dialogues, source_file)
                all_dialogues.extend(processed_dialogues)
                
            elif isinstance(data, dict) and 'scenes' in data:
                # Format: {"scenes": [...]} where each scene has dialogues
                processed_dialogues = []
                for scene in data['scenes']:
                    if 'dialogues' in scene:
                        scene_dialogues = process_standard_dialogues(scene['dialogues'], source_file, scene.get('name', 'Unknown Scene'))
                        processed_dialogues.extend(scene_dialogues)
                all_dialogues.extend(processed_dialogues)
                
            elif isinstance(data, list):
                # List of dialogues directly
                processed_dialogues = process_standard_dialogues(data, source_file)
                all_dialogues.extend(processed_dialogues)
                
            elif isinstance(data, dict) and 'text' in data and isinstance(data['text'], list):
                # Format from page02.json: {"text": [{speaker/action: content}, ...]}
                processed_dialogues = process_text_list_format(data['text'], source_file)
                all_dialogues.extend(processed_dialogues)
                
            else:
                # Try processing any other dict format at the top level
                processed = False
                
                # Check for any list field that might contain dialogues
                for key, value in data.items():
                    if isinstance(value, list) and len(value) > 0:
                        if all(isinstance(item, dict) for item in value):
                            print(f"Found list field '{key}' in {file}, attempting to process...")
                            processed_dialogues = process_text_list_format(value, source_file)
                            if processed_dialogues:
                                all_dialogues.extend(processed_dialogues)
                                processed = True
                                break
                
                if not processed:
                    # Last resort: treat the whole file as a single dialogue entry
                    dialogue_text = json.dumps(data)
                    all_dialogues.append({
                        'source_file': source_file,
                        'dialogue': dialogue_text,
                        'dialogueID': f"{source_file}_whole"
                    })
                    print(f"Processed {file} as a single dialogue entry (fallback method)")
                
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    if not all_dialogues:
        print("No dialogues were extracted from the JSON files")
        return
    
    print(f"Extracted {len(all_dialogues)} dialogues from all files")
    
    # Convert to DataFrame
    dialogues_df = pd.DataFrame(all_dialogues)
    
    # Save all dialogues
    dialogues_path = os.path.join(processed_dir, "raw1_dialogues.csv")
    dialogues_df.to_csv(dialogues_path, index=False)
    print(f"Saved all dialogues to {dialogues_path}")
    
    # Create training examples with prompts
    examples = []
    for _, row in dialogues_df.iterrows():
        source = row['source_file']
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
            
            # Add dialogueID
            example_dict['dialogueID'] = row['dialogueID']
            
            # Add any other metadata columns
            for key in ['scene', 'speaker', 'character', 'context']:
                if key in row and not pd.isna(row[key]):
                    example_dict[key] = row[key]
                    
            examples.append(example_dict)
    
    # Convert to DataFrame and split
    examples_df = pd.DataFrame(examples)
    
    if examples_df.empty:
        print("No valid examples could be created from the dialogues")
        return
    
    # Shuffle data
    examples_df = examples_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into train/val
    train_df, val_df = train_test_split(examples_df, test_size=0.2, random_state=42)
    
    # Save train/val splits
    train_path = os.path.join(processed_dir, "raw1_train.csv")
    val_path = os.path.join(processed_dir, "raw1_val.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"Saved {len(train_df)} training examples to {train_path}")
    print(f"Saved {len(val_df)} validation examples to {val_path}")
    
    # Create formatted versions (you can add additional formatting if needed)
    train_formatted_path = os.path.join(processed_dir, "train_formatted.csv")
    val_formatted_path = os.path.join(processed_dir, "val_formatted.csv")
    
    train_df.to_csv(train_formatted_path, index=False)
    val_df.to_csv(val_formatted_path, index=False)
    
    print(f"Saved formatted training data to {train_formatted_path}")
    print(f"Saved formatted validation data to {val_formatted_path}")
    
    print("\nData processing complete!")
    print(f"CSV files are ready at {processed_dir}")
    
    # Display a sample for verification
    print("\nSample data (first row from training set):")
    for col in ['prompt', 'completion', 'full_text']:
        if col in train_df.columns:
            print(f"\n{col}:")
            sample_text = train_df[col].iloc[0]
            print(sample_text[:150] + "..." if len(sample_text) > 150 else sample_text)

def process_standard_dialogues(dialogues, source_file, scene_name=None):
    """Process dialogues in the standard format."""
    processed_dialogues = []
    
    for i, dialogue in enumerate(dialogues):
        if isinstance(dialogue, dict):
            # If dialogue is a dictionary, extract text or lines
            if 'text' in dialogue:
                dialogue_text = dialogue['text']
            elif 'lines' in dialogue:
                # Join lines if dialogue is split into lines
                lines = dialogue['lines']
                if isinstance(lines, list):
                    dialogue_text = "\n".join([str(line) for line in lines])
                else:
                    dialogue_text = str(lines)
            else:
                # Use the whole dialogue dict minus metadata
                dialogue_copy = dialogue.copy()
                exclude_keys = ['id', 'dialogueID', 'metadata']
                for key in exclude_keys:
                    if key in dialogue_copy:
                        del dialogue_copy[key]
                dialogue_text = json.dumps(dialogue_copy)
        elif isinstance(dialogue, str):
            # If dialogue is directly a string
            dialogue_text = dialogue
        else:
            # Skip if we can't process this dialogue
            print(f"Warning: Skipping dialogue {i} in {source_file} due to unknown format")
            continue
        
        # Create a dialogue entry
        dialogue_entry = {
            'source_file': source_file,
            'dialogue': dialogue_text,
            'dialogueID': dialogue.get('dialogueID', f"{source_file}_{i}") if isinstance(dialogue, dict) else f"{source_file}_{i}"
        }
        
        # Add scene info if available
        if scene_name:
            dialogue_entry['scene'] = scene_name
            
        # Add any other useful metadata from the dialogue
        if isinstance(dialogue, dict):
            for key in ['scene', 'speaker', 'character', 'context']:
                if key in dialogue:
                    dialogue_entry[key] = dialogue[key]
        
        processed_dialogues.append(dialogue_entry)
    
    return processed_dialogues

def process_text_list_format(text_list, source_file):
    """Process the format found in page02.json and similar files."""
    processed_dialogues = []
    
    # Group the dialogue exchanges together
    current_dialogue = []
    dialogue_blocks = []
    
    for item in text_list:
        if isinstance(item, dict):
            # Each item should be a dict with a single key-value pair
            # (speaker/action: text)
            current_dialogue.append(item)
            
            # Check if this is a separator (like {"ACTION": "---"})
            is_separator = False
            for key, value in item.items():
                if key == "ACTION" and (value == "---" or value.startswith("-----")):
                    is_separator = True
                    
            if is_separator and current_dialogue:
                # End of a dialogue section
                if len(current_dialogue) > 1:  # Only save if there's actual dialogue
                    dialogue_blocks.append(current_dialogue[:-1])  # Exclude separator
                current_dialogue = [item]  # Keep separator as start of next block
    
    # Don't forget the last dialogue if it doesn't end with a separator
    if current_dialogue:
        dialogue_blocks.append(current_dialogue)
    
    # Process each dialogue block
    for i, block in enumerate(dialogue_blocks):
        dialogue_lines = []
        
        for item in block:
            for speaker, text in item.items():
                if speaker == "ACTION":
                    dialogue_lines.append(f"[{text}]")
                else:
                    dialogue_lines.append(f"{speaker}: {text}")
        
        dialogue_text = "\n".join(dialogue_lines)
        
        dialogue_entry = {
            'source_file': source_file,
            'dialogue': dialogue_text,
            'dialogueID': f"{source_file}_block_{i}"
        }
        
        processed_dialogues.append(dialogue_entry)
    
    return processed_dialogues

if __name__ == "__main__":
    # Update these paths if your directory structure is different
    process_json_to_csv(
        raw_dir="./data/raw",  # Path to raw JSON files
        processed_dir="./data/processed"  # Path to save processed CSV files
    )