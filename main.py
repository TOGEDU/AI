from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header, Query
import jwt
import json
import base64
from fastapi.responses import FileResponse, StreamingResponse
import uvicorn
from pydantic import BaseModel
from synthesizer import XTTSynthesizer
import os
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# .env 파일 로드
load_dotenv()

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI()

# Base64로 인코딩된 SECRET_KEY를 디코딩
encoded_secret_key = os.getenv("SECRET_KEY")
SECRET_KEY = base64.b64decode(encoded_secret_key)
ALGORITHM = os.getenv("ALGORITHM", "HS512")

# Hugging Face 토큰 가져오기
huggingface_token = os.getenv("HUGGING_KEY")

# Llama3 클래스 정의
class Llama3:
    def __init__(self, model_path, prompts_path):
        self.model_id = model_path
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        # JSON 파일에서 프롬프트 불러오기
        with open(prompts_path, 'r', encoding='utf-8') as f:
            self.prompts = json.load(f)

    def get_response(self, query, message_history=[], max_tokens=4096, temperature=0.6, top_p=0.9):
        # JSON 파일에서 기본 프롬프트를 불러와 사용
        base_prompt = self.prompts.get("parent_response", "")
        user_prompt = message_history + [{"role": "user", "content": query}]
        prompt = base_prompt + " " + self.tokenizer.apply_chat_template(user_prompt, tokenize=False, add_generation_prompt=True)
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        response = outputs[0]["generated_text"][len(prompt):]
        return response, user_prompt + [{"role": "assistant", "content": response}]

# 모델 로드
llama_bot = Llama3("MLP-KTLim/llama-3-Korean-Bllossom-8B", os.path.join(os.path.dirname(__file__), 'app', 'prompts.json'))

# 요청 본문 모델 정의
class TextGenerationRequest(BaseModel):
    input_text: str

async def token_generator(prompt):
    response, _ = llama_bot.get_response(prompt)
    for token in response:
        yield token

@app.post("/generate")
async def generate_text(request: TextGenerationRequest):
    try:
        return StreamingResponse(token_generator(request.input_text), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 기본 엔드포인트
@app.get("/")
def root():
    return {"message": "HI"}

if __name__ == "__main__":
    # FastAPI 서버 실행
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
