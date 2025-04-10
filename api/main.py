from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os
from typing import Optional
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import re

# Start time for uptime tracking
START_TIME = time.time()

app = FastAPI(
    title="Game Dialogue Generator API",
    description="API for generating game dialogue using fine-tuned GPT-2",
    version="1.0.0"
)

# Path to your saved model
MODEL_PATH = "models/dialogue_generator_full"

# Load model and tokenizer - do this at startup
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded successfully on {device}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

class GenerationRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 200
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.92
    no_repeat_ngram_size: Optional[int] = 2

class StatusResponse(BaseModel):
    status: str
    model: str
    uptime: float
    device: str
    version: str

def clean_generated_text(text):
    """Remove metadata markers and other non-dialogue elements."""
    # Remove common metadata patterns
    text = re.sub(r'\[---\]|\[-+\]', '', text)
    text = re.sub(r'Log:.*?[\n\)]', '', text)
    text = re.sub(r'\(This transcript.*?\)', '', text)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

@app.post("/generate")
async def generate_text(request: GenerationRequest):
    try:
        # Prepare inputs
        inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                do_sample=True,
                top_p=request.top_p,
                no_repeat_ngram_size=request.no_repeat_ngram_size
            )
        
        # Decode and clean output
        generated_text = clean_generated_text(tokenizer.decode(outputs[0], skip_special_tokens=True))
        
        # Log the request and response
        print(f"Request: {request.prompt}")
        print(f"Generated: {generated_text[:50]}...")
        
        return {"generated_text": generated_text}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")

@app.get("/status")
async def get_status():
    uptime = time.time() - START_TIME
    return StatusResponse(
        status="running",
        model=MODEL_PATH,
        uptime=uptime, 
        device=device,
        version="1.0.0"
    )

# Add a simple root endpoint
@app.get("/", response_class=FileResponse)
async def get_index():
    return FileResponse(os.path.join(static_dir, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)