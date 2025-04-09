import pytest
from models.generate import generate_content

def test_generate_content():
    prompt = "Create a character backstory for a brave knight."
    output = generate_content(prompt)
    
    assert isinstance(output, str), "Output should be a string."
    assert len(output) > 0, "Output should not be empty."
    assert "knight" in output.lower(), "Output should mention 'knight'."

def test_generate_content_with_empty_prompt():
    prompt = ""
    output = generate_content(prompt)
    
    assert output == "Prompt cannot be empty.", "Output should indicate that the prompt is empty."

def test_generate_content_with_long_prompt():
    prompt = "Generate a detailed lore snippet about an ancient dragon that guards a hidden treasure in a mystical forest."
    output = generate_content(prompt)
    
    assert isinstance(output, str), "Output should be a string."
    assert len(output) > 0, "Output should not be empty."
    assert "dragon" in output.lower(), "Output should mention 'dragon'."