import os
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import time
import uvicorn
from pydantic import BaseModel
import logging
import sys
import time
import pandas as pd

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the generator
from models.generate import GameDialogueGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api/game_dialogue_api.log")
    ]
)
logger = logging.getLogger(__name__)

# Load game names if available
game_names = []
if os.path.exists("data/processed/train_formatted.csv"):
    try:
        train_df = pd.read_csv("data/processed/train_formatted.csv")
        game_names = sorted(train_df['game'].unique().tolist())
        logger.info(f"Loaded {len(game_names)} game names from dataset")
    except Exception as e:
        logger.warning(f"Error loading game names: {e}")

# Load model - check for fine-tuned model first
model_path = "distilgpt2"  # Default
if os.path.exists("models"):
    fine_tuned_models = [d for d in os.listdir("models") if os.path.isdir(os.path.join("models", d))]
    if fine_tuned_models:
        model_path = os.path.join("models", fine_tuned_models[0])
        logger.info(f"Using fine-tuned model: {model_path}")

# Initialize the generator
generator = GameDialogueGenerator(model_path=model_path)
start_time = time.time()

app = FastAPI(title="Video Game Dialogue Generator API", 
             description="API for generating video game dialogue using a fine-tuned GPT-2 model")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class GenerationResponse(BaseModel):
    game: str
    prompt: str
    generated_text: str
    generation_time: float

class StatusResponse(BaseModel):
    status: str
    model_name: str
    uptime: float
    request_count: int
    available_games: list

# Track request count
request_count = 0

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Return the status of the API server"""
    global start_time, request_count
    uptime = time.time() - start_time
    
    return {
        "status": "online",
        "model_name": model_path,
        "uptime": uptime,
        "request_count": request_count,
        "available_games": game_names[:50] if len(game_names) > 50 else game_names  # Limit to 50 games
    }

@app.get("/games")
async def get_games():
    """Return the list of available games"""
    return {"games": game_names}

@app.get("/generate", response_model=GenerationResponse)
async def generate_dialogue(
    game: str = Query(..., description="Game title to generate dialogue for"),
    scene: str = Query(None, description="Optional scene description"),
    max_length: int = Query(200, description="Maximum length of generated text"),
    temperature: float = Query(0.7, description="Generation temperature (0.1-1.0)"),
    top_p: float = Query(0.9, description="Nucleus sampling parameter (0.1-1.0)")
):
    """Generate game dialogue for a specific game"""
    global request_count
    request_count += 1
    
    # Create prompt
    prompt = f"Generate dialogue for a scene in the game '{game}'"
    if scene:
        prompt += f" where {scene}"
    prompt += ":\n"
    
    # Log the request
    logger.info(f"Generation request: game='{game}' scene='{scene}' max_length={max_length} temp={temperature}")
    
    # Validate parameters
    if temperature < 0.1 or temperature > 1.0:
        raise HTTPException(status_code=400, detail="Temperature must be between 0.1 and 1.0")
    if top_p < 0.1 or top_p > 1.0:
        raise HTTPException(status_code=400, detail="top_p must be between 0.1 and 1.0")
    if max_length < 10 or max_length > 1000:
        raise HTTPException(status_code=400, detail="max_length must be between 10 and 1000")
    
    # Generate text
    generation_start = time.time()
    try:
        generated_text = generator.generate(
            prompt, 
            max_length=max_length,
            temperature=temperature,
            top_p=top_p
        )
        generation_time = time.time() - generation_start
        
        # Log success
        logger.info(f"Generation successful: {len(generated_text)} chars in {generation_time:.2f}s")
        
        return {
            "game": game,
            "prompt": prompt,
            "generated_text": generated_text,
            "generation_time": generation_time
        }
    
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# Include web interface
from fastapi.staticfiles import StaticFiles
import shutil

# Create web directory if it doesn't exist
os.makedirs("web", exist_ok=True)

# Create HTML file for the web interface
with open("web/index.html", "w") as f:
    f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Game Dialogue Generator</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="container">
        <h1>Video Game Dialogue Generator</h1>
        <p>Generate authentic dialogue for video games using a fine-tuned language model</p>
        
        <div class="form-group">
            <label for="game">Game Title:</label>
            <select id="game">
                <option value="">Loading games...</option>
            </select>
            <button id="refresh-games">↻</button>
        </div>
        
        <div class="form-group">
            <label for="scene">Scene Description (optional):</label>
            <input type="text" id="scene" placeholder="characters discussing their next quest">
        </div>
        
        <div class="controls">
            <div class="form-group">
                <label for="max-length">Max Length:</label>
                <input type="range" id="max-length" min="50" max="500" value="200">
                <span id="max-length-value">200</span>
            </div>
            
            <div class="form-group">
                <label for="temperature">Temperature:</label>
                <input type="range" id="temperature" min="0.1" max="1.0" step="0.1" value="0.7">
                <span id="temperature-value">0.7</span>
            </div>
        </div>
        
        <button id="generate-btn">Generate Dialogue</button>
        
        <div id="loading" class="hidden">Generating dialogue...</div>
        
        <div id="result" class="hidden">
            <h2>Generated Dialogue</h2>
            <div id="dialogue-container"></div>
            <p id="generation-time"></p>
        </div>
        
        <div class="status">
            <p>API Status: <span id="status">Checking...</span></p>
            <p>Model: <span id="model-name">-</span></p>
            <p>Uptime: <span id="uptime">-</span></p>
            <p>Request Count: <span id="request-count">-</span></p>
        </div>
    </div>
    
    <script src="script.js"></script>
</body>
</html>""")

# Create CSS file
with open("web/style.css", "w") as f:
    f.write("""body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #f5f5f5;
    margin: 0;
    padding: 0;
}

.container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}

h1 {
    color: #2c3e50;
    margin-bottom: 10px;
}

.form-group {
    margin-bottom: 15px;
    position: relative;
}

label {
    display: block;
    margin-bottom: 5px;
    font-weight: bold;
}

input[type="text"], select {
    width: 100%;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 16px;
}

select {
    background-color: white;
}

#refresh-games {
    position: absolute;
    right: 0;
    top: 29px;
    background: #3498db;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 5px 10px;
    cursor: pointer;
}

.controls {
    display: flex;
    gap: 20px;
    margin-bottom: 20px;
}

.controls .form-group {
    flex: 1;
}

button {
    background-color: #3498db;
    color: white;
    border: none;
    padding: 10px 20px;
    cursor: pointer;
    font-size: 16px;
    border-radius: 4px;
    transition: background-color 0.3s;
}

button:hover {
    background-color: #2980b9;
}

#loading {
    margin: 20px 0;
    font-style: italic;
}

#result {
    margin-top: 30px;
    background-color: white;
    padding: 20px;
    border-radius: 5px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

#dialogue-container {
    white-space: pre-wrap;
    font-family: 'Courier New', Courier, monospace;
    background-color: #f9f9f9;
    padding: 15px;
    border-left: 4px solid #3498db;
    margin: 10px 0;
    max-height: 400px;
    overflow-y: auto;
}

#generation-time {
    font-size: 14px;
    color: #7f8c8d;
    text-align: right;
}

.hidden {
    display: none;
}

.status {
    margin-top: 30px;
    padding-top: 20px;
    border-top: 1px solid #ddd;
    font-size: 14px;
    color: #7f8c8d;
}

.status p {
    margin: 5px 0;
}""")

# Create JavaScript file
with open("web/script.js", "w") as f:
    f.write("""document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const gameSelect = document.getElementById('game');
    const sceneInput = document.getElementById('scene');
    const maxLengthSlider = document.getElementById('max-length');
    const maxLengthValue = document.getElementById('max-length-value');
    const temperatureSlider = document.getElementById('temperature');
    const temperatureValue = document.getElementById('temperature-value');
    const generateBtn = document.getElementById('generate-btn');
    const refreshGamesBtn = document.getElementById('refresh-games');
    const loadingDiv = document.getElementById('loading');
    const resultDiv = document.getElementById('result');
    const dialogueContainer = document.getElementById('dialogue-container');
    const generationTime = document.getElementById('generation-time');
    
    // Status elements
    const statusSpan = document.getElementById('status');
    const modelNameSpan = document.getElementById('model-name');
    const uptimeSpan = document.getElementById('uptime');
    const requestCountSpan = document.getElementById('request-count');
    
    // Update slider values
    maxLengthSlider.addEventListener('input', () => {
        maxLengthValue.textContent = maxLengthSlider.value;
    });
    
    temperatureSlider.addEventListener('input', () => {
        temperatureValue.textContent = temperatureSlider.value;
    });
    
    // Load games
    async function loadGames() {
        try {
            const response = await fetch('/games');
            if (response.ok) {
                const data = await response.json();
                
                // Clear the select
                gameSelect.innerHTML = '';
                
                // Add games as options
                if (data.games && data.games.length > 0) {
                    data.games.forEach(game => {
                        const option = document.createElement('option');
                        option.value = game;
                        option.textContent = game;
                        gameSelect.appendChild(option);
                    });
                } else {
                    // Add a default option if no games
                    const option = document.createElement('option');
                    option.value = "Final Fantasy VII";
                    option.textContent = "Final Fantasy VII";
                    gameSelect.appendChild(option);
                }
            } else {
                throw new Error(`API error: ${response.status}`);
            }
        } catch (error) {
            console.error("Error loading games:", error);
            // Add a default option
            gameSelect.innerHTML = '';
            const option = document.createElement('option');
            option.value = "Final Fantasy VII";
            option.textContent = "Final Fantasy VII";
            gameSelect.appendChild(option);
        }
    }
    
    // Refresh games button
    refreshGamesBtn.addEventListener('click', loadGames);
    
    // Generate button click handler
    generateBtn.addEventListener('click', async () => {
        const game = gameSelect.value;
        const scene = sceneInput.value.trim();
        
        if (!game) {
            alert('Please select a game');
            return;
        }
        
        // Show loading, hide result
        loadingDiv.classList.remove('hidden');
        resultDiv.classList.add('hidden');
        generateBtn.disabled = true;
        
        try {
            // Build query parameters
            let url = `/generate?game=${encodeURIComponent(game)}&max_length=${maxLengthSlider.value}&temperature=${temperatureSlider.value}&top_p=0.9`;
            if (scene) {
                url += `&scene=${encodeURIComponent(scene)}`;
            }
            
            // Call API
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Display result
            dialogueContainer.textContent = data.generated_text;
            generationTime.textContent = `Generated in ${data.generation_time.toFixed(2)} seconds`;
            
            resultDiv.classList.remove('hidden');
        } catch (error) {
            alert(`Error: ${error.message}`);
            console.error(error);
        } finally {
            loadingDiv.classList.add('hidden');
            generateBtn.disabled = false;
        }
    });
    
    // Function to format uptime
    function formatUptime(seconds) {
        const days = Math.floor(seconds / 86400);
        const hours = Math.floor((seconds % 86400) / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        let result = '';
        if (days > 0) result += `${days}d `;
        if (hours > 0) result += `${hours}h `;
        if (minutes > 0) result += `${minutes}m `;
        result += `${secs}s`;
        
        return result;
    }
    
    // Check API status
    async function checkStatus() {
        try {
            const response = await fetch('/status');
            if (response.ok) {
                const data = await response.json();
                statusSpan.textContent = data.status;
                modelNameSpan.textContent = data.model_name;
                uptimeSpan.textContent = formatUptime(data.uptime);
                requestCountSpan.textContent = data.request_count;
                statusSpan.style.color = 'green';
                
                // If we don't have games loaded yet, use the ones from status
                if (gameSelect.options.length <= 1 && data.available_games && data.available_games.length > 0) {
                    gameSelect.innerHTML = '';
                    data.available_games.forEach(game => {
                        const option = document.createElement('option');
                        option.value = game;
                        option.textContent = game;
                        gameSelect.appendChild(option);
                    });
                }
            } else {
                statusSpan.textContent = 'Error';
                statusSpan.style.color = 'red';
            }
        } catch (error) {
            statusSpan.textContent = 'Offline';
            statusSpan.style.color = 'red';
        }
    }
    
    // Initialize
    loadGames();
    checkStatus();
    
    // Check status every 30 seconds
    setInterval(checkStatus, 30000);
});""")

# Mount static files
app.mount("/", StaticFiles(directory="web", html=True), name="web")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)