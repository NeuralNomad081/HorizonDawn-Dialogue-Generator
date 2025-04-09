import os
import json
import pandas as pd

def check_json_files(directory='data/raw'):
    """Check all JSON files in the given directory and report their structure."""
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return
        
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in {directory}")
        return
        
    print(f"Found {len(json_files)} JSON files in {directory}")
    
    for file in json_files:
        file_path = os.path.join(directory, file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Print basic info about the JSON structure
            print(f"\n--- {file} ---")
            if isinstance(data, list):
                print(f"List with {len(data)} items")
                if data:  # If list is not empty
                    print(f"First item keys: {data[0].keys() if isinstance(data[0], dict) else 'Not a dictionary'}")
            elif isinstance(data, dict):
                print(f"Dictionary with {len(data)} keys")
                print(f"Keys: {data.keys()}")
            else:
                print(f"Unexpected type: {type(data)}")
        except json.JSONDecodeError:
            print(f"Error: {file} contains invalid JSON")
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")

if __name__ == "__main__":
    check_json_files()