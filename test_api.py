"""
Test script for the Llama API
"""
import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8000"  # Change to your droplet IP in production

def test_health():
    """Test health endpoint"""
    print("🏥 Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_root():
    """Test root endpoint"""
    print("🏠 Testing root endpoint...")
    response = requests.get(BASE_URL)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_generate(prompt, **kwargs):
    """Test generation endpoint"""
    print(f"🤖 Testing generation with prompt: '{prompt}'")
    
    payload = {"prompt": prompt, **kwargs}
    
    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/generate",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    elapsed = time.time() - start_time
    
    print(f"Status: {response.status_code}")
    print(f"Time: {elapsed:.2f}s")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Generated text:\n{result['generated_text']}")
        print(f"\nTokens - Prompt: {result['prompt_tokens']}, Generated: {result['generated_tokens']}")
    else:
        print(f"Error: {response.text}")
    print("-" * 80 + "\n")

def test_stats():
    """Test stats endpoint"""
    print("📊 Testing stats endpoint...")
    response = requests.get(f"{BASE_URL}/stats")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

if __name__ == "__main__":
    print("=" * 80)
    print("Llama-3.2-1B API Test Suite")
    print("=" * 80 + "\n")
    
    try:
        # Test basic endpoints
        test_health()
        test_root()
        test_stats()
        
        # Test generation with different prompts
        test_generate(
            "The key to life is",
            max_new_tokens=100,
            temperature=0.7
        )
        
        test_generate(
            "Explain quantum computing in simple terms:",
            max_new_tokens=150,
            temperature=0.8
        )
        
        test_generate(
            "Write a haiku about technology",
            max_new_tokens=50,
            temperature=0.9
        )
        
        print("✅ All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API.")
        print("Make sure the service is running with: docker-compose up -d")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
