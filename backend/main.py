import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "qwen-finetuned")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
).eval()

class Message(BaseModel):
    role: str 
    content: str

class ChatRequest(BaseModel):
    session_id: str
    messages: List[Message]
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", 200))

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint for processing user messages and generating AI responses."""
    try:
        formatted_history = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        text = tokenizer.apply_chat_template(
            formatted_history,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=request.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.3,
            top_p=0.1,
            repetition_penalty=1.3,
            length_penalty=0.8,
            no_repeat_ngram_size=4,
            early_stopping=True
        )
        
        new_tokens = outputs[0][inputs.input_ids.shape[-1]:]
        response_content = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        response_content = response_content.split("<|im_end|>")[0].strip()
        
        response_messages = request.messages.copy()
        response_messages.append(Message(role="assistant", content=response_content))
        
        return {"messages": response_messages}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint to verify if the service is running."""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
