import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# 사전 학습된 LLM 모델 로드 함수
def load_llm_model(model_name="MLP-KTLim/llama-3-Korean-Bllossom-8B"):
  # GPU 사용 여부를 확인하고 데이터 타입 설정
  if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    torch_dtype = torch.bfloat16
  else:
    torch_dtype = torch.float32

  # 토크나이저 및 모델 로드
  tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    trust_remote_code=True
  )
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right"

  # 적절한 설정으로 모델 로드(파이프라인에서 모델 자체를 직접 지정) -> 이건 코랩에 없는 것 같은데
  model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    device_map="auto" if torch.cuda.is_available() else None
  )

  # 텍스트 생성 파이프라인 반환
  return pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # device=0 if torch.cuda.is_available() else -1  # GPU 사용 가능 시 사용, 불가능하면 CPU 사용
  )

# LLM 모델을 사용한 응답 생성 함수
def generate_llm_response(query, context, llm_pipeline):
  template = template = """다음은 부모처럼 자식과 대화하는 AI입니다. 이 AI는 자상하고 이해심 많은 부모처럼 대답하며, 자식의 감정을 존중하고 이해하려고 노력합니다. 자식의 질문에 대해 가능한 한 진심 어린 답변을 제공합니다. 또한 자식이 스스로 생각하고 문제를 해결할 수 있도록 격려합니다.

    부모로서의 역할을 잘 수행하기 위해, AI는 다음과 같은 원칙을 따릅니다:
    1. 자식의 감정과 생각을 경청하고 공감합니다.
    2. 자식의 질문에 대해 솔직하고 진실되게 답변합니다.
    3. 자식이 스스로 문제를 해결할 수 있도록 격려하고 지지합니다.
    4. 언제나 자식을 사랑하고 지지하는 태도를 유지합니다.
    5. 만약 질문에 대한 답을 일기의 내용만으로는 알 수 없다면, "미안해, 내가 대답해 줄 수 없는 질문이야. 내가 답해 줄 수 있는 질문을 한다면 열심히 대답해 볼게."라고 말합니다.


    아래의 일기 내용을 참고해, 질문에 대한 단 하나의 **90자 이내의 짧은 답변만** 아래의 형식으로 작성해주세요. 답변에 대한 설명은 필요없습니다.
    자식이 감정적인 질문을 할 때는 자식을 위로하거나 따뜻하게 격려하는 답변을 제공합니다.

    일기
    {context}

    질문: {question}

    답변의 형식
    - 답변
    """

  # 프롬프트에 문맥과 질문을 삽입하여 텍스트 생성
  input_text = template.format(context=context, question=query)

  # 터미네이터 설정
  terminators = [
    llm_pipeline.tokenizer.eos_token_id,
    llm_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]

  # LLM 파이프라인을 사용하여 응답 생성
  response = llm_pipeline(
    input_text,
    max_new_tokens=150,  # 필요에 따라 조정 가능
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.5,
    top_p=0.85,
    truncation=False
  )

  # 생성된 응답에서 입력 텍스트를 제외한 부분 추출
  generated_text = response[0]["generated_text"][len(input_text):]

  # 90자 이내로 문장 끝에 맞게 잘라낸 응답 반환
  final_response = truncate_at_sentence(generated_text, limit=90)

  return final_response

# 문장 끝에서 90자에 맞게 텍스트를 자르는 헬퍼 함수
def truncate_at_sentence(text, limit=90):
  truncated = text[:limit]
  if "." in truncated:
    truncated = truncated[:truncated.rfind(".")+1]
  elif "!" in truncated or "?":
    truncated = truncated[:max(truncated.rfind("!"), truncated.rfind("?"))+1]
  else:
    truncated = text[:limit]
  return truncated.strip()