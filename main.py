"""
FastAPI web service for Llama-3.2-1B
Optimized for 4GB RAM DigitalOcean droplet
"""
import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Llama-3.2-1B API",
    description="Optimized LLM inference API for DigitalOcean",
    version="1.0.0"
)

# Global variables for model and tokenizer
model = None
tokenizer = None

# Configuration for 4-bit quantization (saves ~75% memory)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

class TextGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt", min_length=1, max_length=2048)
    max_new_tokens: Optional[int] = Field(default=256, ge=1, le=1024)
    temperature: Optional[float] = Field(default=0.7, ge=0.1, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.1, le=1.0)
    do_sample: Optional[bool] = Field(default=True)

class TextGenerationResponse(BaseModel):
    generated_text: str
    prompt_tokens: int
    generated_tokens: int

@app.on_event("startup")
async def load_model():
    """Load model and tokenizer on startup with memory optimization"""
    global model, tokenizer
    
    try:
        logger.info("Loading Llama-3.2-1B model with 4-bit quantization...")
        model_id = "meta-llama/Llama-3.2-1B"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        
        # Load model with 4-bit quantization for low memory usage
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        logger.info("Model loaded successfully!")
        logger.info(f"Model device: {model.device}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "Llama-3.2-1B API",
        "model": "meta-llama/Llama-3.2-1B",
        "optimization": "4-bit quantization",
        "endpoints": {
            "generate": "/generate",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.post("/generate", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    """Generate text from the model"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Tokenize input
        inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
        prompt_tokens = inputs.input_ids.shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.do_sample,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_tokens = outputs.shape[1] - prompt_tokens
        
        return TextGenerationResponse(
            generated_text=generated_text,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get model and system statistics"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get memory stats if CUDA is available
        memory_stats = {}
        if torch.cuda.is_available():
            memory_stats = {
                "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
                "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2)
            }
        
        return {
            "model_name": "meta-llama/Llama-3.2-1B",
            "quantization": "4-bit",
            "device": str(model.device),
            "memory": memory_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
