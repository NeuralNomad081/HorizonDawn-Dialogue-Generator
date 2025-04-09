import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_generate_endpoint():
    response = client.get("/generate?prompt=Test prompt")
    assert response.status_code == 200
    assert "generated_content" in response.json()

def test_status_endpoint():
    response = client.get("/status")
    assert response.status_code == 200
    assert "uptime" in response.json()