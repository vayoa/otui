import json
import uuid
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from groq_brains import GroqBrain, Message
from brains import Brain

CHAT_DIR = Path("chats")
CHAT_DIR.mkdir(exist_ok=True)

class Chat:
    def __init__(self):
        self.brain = GroqBrain()
        if not self.brain.messages:
            self.brain.add_messages([
                {"role": "system", "content": "You are a helpful assistant."}
            ])

    def messages(self) -> List[Message]:
        return self.brain.messages

    def save(self, chat_id: str):
        with open(CHAT_DIR / f"{chat_id}.json", "w") as f:
            json.dump(self.brain.messages, f)

    @staticmethod
    def load(chat_id: str) -> "Chat":
        chat = Chat()
        file = CHAT_DIR / f"{chat_id}.json"
        if file.exists():
            chat.brain.set_messages(json.load(open(file)))
        return chat

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

chats: Dict[str, Chat] = {}

class SendRequest(BaseModel):
    content: str

class EditRequest(BaseModel):
    index: int
    content: str

@app.post("/api/chats")
def create_chat():
    chat_id = str(uuid.uuid4())
    chats[chat_id] = Chat()
    return {"id": chat_id}

@app.get("/api/chats")
def list_chats():
    return [{"id": cid} for cid in chats.keys()]

@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: str):
    chat = chats.get(chat_id)
    if not chat:
        raise HTTPException(status_code=404)
    return {"messages": chat.messages()}

@app.post("/api/chats/{chat_id}/messages")
def send_message(chat_id: str, req: SendRequest):
    chat = chats.get(chat_id)
    if not chat:
        raise HTTPException(status_code=404)
    chat.brain.add_messages([{"role": "user", "content": req.content}])
    answer = ""
    for chunk in chat.brain.chat(input=req.content, stream=True):
        delta = chunk.choices[0].delta
        answer += delta.content or ""
    chat.brain.add_messages([{"role": "assistant", "content": answer}])
    chat.save(chat_id)
    return {"content": answer}

@app.post("/api/chats/{chat_id}/edit")
def edit_message(chat_id: str, req: EditRequest):
    chat = chats.get(chat_id)
    if not chat:
        raise HTTPException(status_code=404)
    chat.brain.update_message_content(req.content, req.index)
    chat.save(chat_id)
    return {"status": "ok"}

@app.post("/api/chats/{chat_id}/regenerate")
def regenerate(chat_id: str):
    chat = chats.get(chat_id)
    if not chat:
        raise HTTPException(status_code=404)
    if len(chat.brain.messages) < 2:
        raise HTTPException(status_code=400)
    user_msg = chat.brain.messages[-2]["content"]
    chat.brain.clear_last_messages(1)
    answer = ""
    for chunk in chat.brain.chat(input=user_msg, stream=True):
        delta = chunk.choices[0].delta
        answer += delta.content or ""
    chat.brain.add_messages([{"role": "assistant", "content": answer}])
    chat.save(chat_id)
    return {"content": answer}
