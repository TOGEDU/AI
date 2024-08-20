from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
#hugginface로그인하고 token입력
import huggingface_hub
hugging_key = os.getenv("HUGGING_KEY")
huggingface_hub.login(hugging_key)

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_docs(documents,chunk_size=1000,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

model_name = "jhgan/ko-sroberta-multitask" # (KorNLU 데이터셋에 학습시킨 한국어 임베딩 모델)
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

def rag(question):
    # 텍스트 파일을 직접 열어서 utf-8 인코딩으로 읽기
    file_path = r"C:\Users\gram\Desktop\AI\llm\a.txt"

    with open(file_path, 'r', encoding='utf-8') as f:
      text = f.read()
#처음 실행할때
    # 텍스트를 Document 객체로 변환
    documents = [Document(page_content=text)]
    # docs 변수에 분할 문서를 저장
    docs = split_docs(documents)
    db = Chroma.from_documents(documents=docs, embedding=embedding_model)
    retriever = db.as_retriever(search_type="mmr")
    docs = db.similarity_search_with_score(question)
    #docs = retriever.get_relevant_documents(question)
    #Chroma.from_documents(docs, embedding_model, persist_directory="./chroma_db")
#2번째 실행할때
    #db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
  
    # 유사도가 가장 높은 결과를 선택
    best_doc, best_score = max(docs, key=lambda x: x[1])
    # 결과 출력
    print(best_doc.page_content)
    #전체 결과 다 보기
    #for idx, doc in enumerate(docs):
    #    print(idx, doc)

    
    
    

