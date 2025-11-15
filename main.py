"""
FastAPI web service for Llama-3.2-1B
Heavily optimized for CPU-only deployment on DigitalOcean droplet
"""
import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Aggressive CPU optimizations
os.environ["OMP_NUM_THREADS"] = "4"  # OpenMP threads
os.environ["MKL_NUM_THREADS"] = "4"  # MKL threads
torch.set_num_threads(4)  # PyTorch threads (use all cores on 2vCPU)
torch.set_grad_enabled(False)  # Disable gradients globally
torch.backends.quantized.engine = 'qnnpack'  # Optimized backend for ARM/x86

# Initialize FastAPI app
app = FastAPI(
    title="Llama-3.2-1B API",
    description="CPU-Optimized LLM inference API",
    version="1.0.0"
)

# Global variables
model = None
tokenizer = None
generation_config = None

class TextGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt", min_length=1, max_length=2048)
    max_new_tokens: Optional[int] = Field(default=100, ge=1, le=512)  # Reduced default
    temperature: Optional[float] = Field(default=0.7, ge=0.1, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.1, le=1.0)
    do_sample: Optional[bool] = Field(default=True)
    # New fields for speed
    top_k: Optional[int] = Field(default=50, ge=1, le=100)  # Limit sampling space

class TextGenerationResponse(BaseModel):
    generated_text: str
    prompt_tokens: int
    generated_tokens: int

@app.on_event("startup")
async def load_model():
    """Load model and tokenizer with aggressive CPU optimizations"""
    global model, tokenizer, generation_config
    
    try:
        logger.info("Loading Llama-3.2-1B model for CPU inference...")
        model_id = "meta-llama/Llama-3.2-1B"
        
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.error("HF_TOKEN environment variable not set!")
            raise ValueError("HF_TOKEN required")
        
        # Load tokenizer with padding
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=hf_token,
            trust_remote_code=True,
            padding_side="left"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with aggressive optimizations
        logger.info("Loading model (this may take a few minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            torch_dtype=torch.float32,  # float32 for CPU (better compatibility)
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Set to eval mode and freeze
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        # Create optimized generation config
        generation_config = GenerationConfig(
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,  # Critical for speed
            num_beams=1,  # Greedy/sampling only
        )
        
        logger.info("Model loaded successfully!")
        logger.info(f"Model device: {model.device}")
        logger.info(f"Model dtype: {model.dtype}")
        
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
        "optimization": "CPU-optimized (float32, KV cache, inference_mode)",
        "endpoints": {
            "generate": "/generate",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.post("/generate", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    """Generate text with maximum CPU optimization"""
    global model, tokenizer, generation_config
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Tokenize with minimal overhead
        inputs = tokenizer(
            request.prompt, 
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512  # Limit input size
        )
        prompt_tokens = inputs.input_ids.shape[1]
        
        # Update generation config with request parameters
        gen_config = generation_config
        gen_config.max_new_tokens = min(request.max_new_tokens, 200)  # Hard cap
        gen_config.temperature = request.temperature
        gen_config.top_p = request.top_p
        gen_config.top_k = request.top_k
        gen_config.do_sample = request.do_sample
        
        # Generate with maximum optimization
        with torch.inference_mode():  # More aggressive than no_grad
            outputs = model.generate(
                inputs.input_ids,
                generation_config=gen_config,
                attention_mask=inputs.attention_mask if 'attention_mask' in inputs else None
            )
        
        # Decode efficiently
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
