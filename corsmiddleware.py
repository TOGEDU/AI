from fastapi.middleware.cors import CORSMiddleware

def apply_cors_middleware(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 또는 ["*"]으로 모든 도메인 허용 배포 시 변경 필요
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )