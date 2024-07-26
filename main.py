from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import uvicorn
from pydantic import BaseModel
from synthesizer import XTTSynthesizer
import os

app = FastAPI()

class SynthesizeRequest(BaseModel):
    config_path: str
    tokenizer_path: str
    checkpoint_path: str
    speaker_reference_path: str
    text: str

def remove_file(file_path: str):
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"Error removing file {file_path}: {e}")

@app.post("/synthesize")
def synthesize(request: SynthesizeRequest, background_tasks: BackgroundTasks):
    try:
        print(f"Config Path: {request.config_path}")
        print(f"Tokenizer Path: {request.tokenizer_path}")
        print(f"Checkpoint Path: {request.checkpoint_path}")
        print(f"Speaker Reference Path: {request.speaker_reference_path}")

        # 파일 경로 존재 여부 확인
        if not os.path.exists(request.config_path):
            raise HTTPException(status_code=400, detail=f"Config file not found: {request.config_path}")
        if not os.path.exists(request.tokenizer_path):
            raise HTTPException(status_code=400, detail=f"Tokenizer file not found: {request.tokenizer_path}")
        if not os.path.exists(request.checkpoint_path):
            raise HTTPException(status_code=400, detail=f"Checkpoint file not found: {request.checkpoint_path}")
        if not os.path.exists(request.speaker_reference_path):
            raise HTTPException(status_code=400, detail=f"Speaker reference file not found: {request.speaker_reference_path}")

        synthesizer = XTTSynthesizer(
            config_path=request.config_path,
            tokenizer_path=request.tokenizer_path,
            checkpoint_path=request.checkpoint_path,
            speaker_reference_path=request.speaker_reference_path,
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


@app.get("/")
def root():
    return {"message":"HI"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
