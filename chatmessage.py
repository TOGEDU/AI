from pydantic import BaseModel

class ChatMessage(BaseModel):
    id: int
    message: str
    chat_room_id: int