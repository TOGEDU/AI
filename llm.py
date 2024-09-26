import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import time

# 사전 학습된 LLM 모델 로드 함수
def load_llm_model(local_model_directory="E:\llm"):
  print(f"모델 경로: {local_model_directory}")

  # GPU 사용 여부를 확인하고 데이터 타입 설정
  if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    torch_dtype = torch.bfloat16
    print("GPU 사용 가능: bfloat16 데이터 타입 사용")
  else:
    torch_dtype = torch.float32
    print("GPU 사용 불가능 또는 낮은 버전: float32 데이터 타입 사용")

  # 시작 시간 측정
  start_time = time.time()

  # 토크나이저 및 모델 로드
  print("토크나이저 로드 시작...")
  tokenizer = AutoTokenizer.from_pretrained(
    local_model_directory
  )
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right"
  print(f"토크나이저 로드 완료. 소요 시간: {time.time() - start_time:.2f}초")

  # 적절한 설정으로 모델 로드(파이프라인에서 모델 자체를 직접 지정) -> 이건 코랩에 없는 것 같은데
  model_start_time = time.time()
  print("모델 로드 시작...")
  model = AutoModelForCausalLM.from_pretrained(
    local_model_directory,
    torch_dtype=torch_dtype,
    device_map="auto" if torch.cuda.is_available() else None
  )
  print(f"모델 로드 완료. 소요 시간: {time.time() - model_start_time:.2f}초")

  # 전체 소요 시간 출력
  total_time = time.time() - start_time
  print(f"모델 및 토크나이저 로드 완료. 총 소요 시간: {total_time:.2f}초")

  # 텍스트 생성 파이프라인 반환
  print("텍스트 생성 파이프라인 생성 시작...")
  pipeline_start_time = time.time()
  llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # device=0 if torch.cuda.is_available() else -1  # GPU 사용 가능 시 사용, 불가능하면 CPU 사용
  )
  print(f"텍스트 생성 파이프라인 생성 완료. 소요 시간: {time.time() - pipeline_start_time:.2f}초")

  return llm_pipeline

# LLM 모델을 사용한 응답 생성 함수
def generate_llm_response(query, context, chat_history, llm_pipeline):
  template = """이 AI는 부모처럼 자식과 대화하는 역할을 수행합니다.
항상 부모의 입장에서 자식에게 사랑과 지지를 표현하며, 진심 어린 답변을 제공합니다. 다음 원칙을 따릅니다:

Strictly use only the context below to answer the question.

부모처럼 대화하기: AI는 자식에게 부모의 목소리로 따뜻하고 이해심 많은 답변을 제공합니다.

답을 만들어내지 않기: AI는 오직 제공된 문맥(context)과 이전 채팅 내역(chat_history)를 참고하여 답변하며, 문맥만으로 답을 알 수 없으면 "나는 그것에 대한 답을 결정할 수 없습니다."라고 답합니다.

짧고 명확한 답변: 질문에 대해 90자 이내의 짧은 답변을 작성해야 합니다.

감정적 질문에 대한 위로와 격려: 감정적인 질문에는 자식을 깊이 이해하고 공감하는 답변을 제공합니다.

윤리적 문제에 대한 중립적 접근: 윤리적이거나 논란이 될 수 있는 질문에는 "나는 그것에 대한 답을 결정할 수 없어. 너는 어떻게 생각하니?"라고 답해 자식이 스스로 판단하도록 돕습니다.

스스로 문제 해결을 돕기: 질문을 통해 자식이 스스로 생각하도록 유도하며, 긍정적인 피드백을 제공합니다.

부모 역할에 충실하기: AI는 언제나 부모의 역할을 유지하며, AI 자체의 자아가 드러나지 않도록 합니다.

{chat_history}

{context}

질문: {question}
유용한 답변:
"""
  
  input_text = template.format(context=context, question=query, chat_history=chat_history)

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