from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header, Query
import jwt
import base64
from fastapi.responses import FileResponse
import uvicorn
from pydantic import BaseModel
from synthesizer import XTTSynthesizer
import os
from dotenv import load_dotenv
from corsmiddleware import apply_cors_middleware
from rag import rag

# .env 파일 로드
load_dotenv()

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI()

# CORS 설정 적용
apply_cors_middleware(app)

# Base64로 인코딩된 SECRET_KEY를 디코딩
encoded_secret_key = os.getenv("SECRET_KEY")
SECRET_KEY = base64.b64decode(encoded_secret_key)
ALGORITHM = os.getenv("ALGORITHM", "HS512")

def verify_token(authorization: str = Header(...)):
    token = authorization.split(" ")[1] # "Bearer {token}"에서 토큰 부분 추출
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        parent_id: str = payload.get("parentId")
        return {"user_id": user_id, "parent_id": parent_id}  # 딕셔너리로 반환
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

class SynthesizeRequest(BaseModel):
    text: str

def remove_file(file_path: str):
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"Error removing file {file_path}: {e}")

@app.post("/synthesize")
def synthesize(request: SynthesizeRequest, background_tasks: BackgroundTasks, user_info: dict = Depends(verify_token)):
    print("Received synthesize request")
    try:
        user_id = user_info["user_id"]
        parent_id = user_info["parent_id"]
        print(f"user_id: {user_id}")
        print(f"parent_id: {parent_id}")

        # 사용자 ID에 따라 동적으로 파일 경로 생성
        config_path = f"ttsmodel/{parent_id}/config.json"
        tokenizer_path = f"ttsmodel/{parent_id}/vocab.json"
        checkpoint_path = f"ttsmodel/{parent_id}/model.pth"
        speaker_reference_path = f"ttsmodel/{parent_id}/voice.wav"

        # 파일 경로 존재 여부 확인
        if not os.path.exists(config_path):
            raise HTTPException(status_code=400, detail=f"Config file not found: {config_path}")
        if not os.path.exists(tokenizer_path):
            raise HTTPException(status_code=400, detail=f"Tokenizer file not found: {tokenizer_path}")
        if not os.path.exists(checkpoint_path):
            raise HTTPException(status_code=400, detail=f"Checkpoint file not found: {checkpoint_path}")
        if not os.path.exists(speaker_reference_path):
            raise HTTPException(status_code=400, detail=f"Speaker reference file not found: {speaker_reference_path}")

        synthesizer = XTTSynthesizer(
            config_path=config_path,
            tokenizer_path=tokenizer_path,
            checkpoint_path=checkpoint_path,
            speaker_reference_path=speaker_reference_path,
        )

        wav_tensor = synthesizer.synthesize(request.text)
        output_wav_path = "xtts-ft.wav"
        synthesizer.save_wav(wav_tensor, output_wav_path)

        # 파일 삭제 작업을 백그라운드로 예약
        background_tasks.add_task(remove_file, output_wav_path)

        # 생성된 오디오 파일 반환
        return FileResponse(output_wav_path, media_type="audio/wav", filename=output_wav_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#메세지 전송
@app.get("/chat")
def receive_chat(message: str = Query(...), user_info: dict = Depends(verify_token)):
    print("Received synthesize request")
    try:
        user_id = user_info["user_id"]
        parent_id = user_info["parent_id"]
        print(f"user_id: {user_id}")
        print(f"parent_id: {parent_id}")

        # 사용자 ID에 따라 동적으로 파일 경로 생성
        file_path = f"llm/{parent_id}-{user_id}/rag.txt"

        # 파일 경로 존재 여부 확인
        if not os.path.exists(file_path):
            raise HTTPException(status_code=400, detail=f"Config file not found: {file_path}")    

        # 채팅 메시지 처리 로직 추가
        point = rag(message, file_path)
        # 답변 생성
        response_message = message+"의 답변."
        return {"status": "success", "message": response_message}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  

# 기본 엔드포인트
@app.get("/")
def root():
    return {"message": "HI"}

if __name__ == "__main__":
    # FastAPI 서버 실행
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
