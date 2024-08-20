from fastapi import FastAPI, HTTPException
import uvicorn
from dotenv import load_dotenv
from app.main import app as app_module

# .env 파일 로드
load_dotenv()

# FastAPI 애플리케이션 인스턴스 사용
app = app_module

if __name__ == "__main__":
    # FastAPI 서버 실행
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
