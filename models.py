from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from database import Base

class ChatRoom(Base):
  __tablename__ = 'chat_room'

  id = Column(Integer, primary_key=True, index=True)
  child_id = Column(Integer)
  date = Column(DateTime)
  summary = Column(String(50), nullable=True)
  chat_messages = relationship("ChatMessage", back_populates="chatroom")

class ChatMessage(Base):
  __tablename__ = 'chat_message'

  id = Column(Integer, primary_key=True, index=True)
  chat_room_id = Column(Integer, ForeignKey('chat_room.id'))
  role = Column(Integer)
  time = Column(DateTime)
  message = Column(String(500))

  chatroom = relationship("ChatRoom", back_populates="chat_messages")