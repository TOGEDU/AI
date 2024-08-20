import json
import os
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
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