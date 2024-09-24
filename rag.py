from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
import huggingface_hub
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch

# .env 파일 로드
load_dotenv()

# hugginface로그인하고 token입력
hugging_key = os.getenv("HUGGING_KEY")
if hugging_key is not None:
  huggingface_hub.login(hugging_key)
else:
  raise ValueError("Hugging Face API key not found in environment variables.")

# 텍스트를 chunk로 분할하는 함수
def split_docs(text, chunk_size=1000, chunk_overlap=200):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  splits = text_splitter.split_text(text)  # split_documents 대신 split_text 사용
  return splits

# 임베딩 모델 설정
model_name = "jhgan/ko-sroberta-multitask" # (KorNLU 데이터셋에 학습시킨 한국어 임베딩 모델)
# GPU가 사용 가능한지 확인
device = 'cuda' if torch.cuda.is_available() else 'cpu'
encode_kwargs = {'normalize_embeddings': False}

embedding_model = HuggingFaceEmbeddings(
  model_name=model_name,
  model_kwargs={'device': device}, # 자동으로 'cuda' 또는 'cpu' 설정
  encode_kwargs=encode_kwargs
)

# RAG 기술을 활용한 context 검색
def rag(question, file_path):
  # 텍스트 파일을 직접 열어서 utf-8 인코딩으로 읽기
  if not os.path.exists(file_path):
    raise FileNotFoundError(f"File {file_path} not found.")
  
  with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()
      
  # 텍스트를 분할하여 splits에 저장
  splits = split_docs(text)

  # Document 객체 생성
  documents = [Document(page_content=split) for split in splits]
  # FAISS 벡터 스토어에 임베딩된 문서 저장
  db = FAISS.from_documents(documents=documents, embedding=embedding_model)
  # 검색을 위한 retriever 생성
  retriever = db.as_retriever(search_type="mmr")
  # 질문에 맞는 문서 검색
  retriever_docs = retriever.get_relevant_documents(question)
  # 검색된 문서를 context로 합치기
  context = "\n\n".join([doc.page_content for doc in retriever_docs])
  # 결과 출력
  print(context)

  return context