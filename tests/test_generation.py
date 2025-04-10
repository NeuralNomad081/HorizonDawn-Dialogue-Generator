import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.generate import generate_dialogue 
def test_generate_dialogue():
    prompt = "Create a character backstory for a brave knight."
    output = generate_dialogue(prompt)
    
    assert isinstance(output, str), "Output should be a string."
    assert len(output) > 0, "Output should not be empty."
    assert "knight" in output.lower(), "Output should mention 'knight'."

def test_generate_content_with_empty_prompt():
    prompt = ""
    output = generate_dialogue(prompt)
    
    assert output == "Prompt cannot be empty.", "Output should indicate that the prompt is empty."

def test_generate_content_with_long_prompt():
    prompt = "Generate a detailed lore snippet about an ancient dragon that guards a hidden treasure in a mystical forest."
    output = generate_dialogue(prompt)
    
    assert isinstance(output, str), "Output should be a string."
    assert len(output) > 0, "Output should not be empty."
    assert "dragon" in output.lower(), "Output should mention 'dragon'."
    
    
    
'''
TO run tests
cd /{project directory}
python -m pytest tests/test_generation.py -v

'''