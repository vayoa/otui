from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import os
from typing import Optional
from datetime import datetime

from ui import _DEFAULT_TIME_FORMAT


def create_app(ui) -> FastAPI:
    app = FastAPI()

    static_path = "web/main page"
    app.mount("/static", StaticFiles(directory=static_path), name="static")

    @app.get("/chats")
    async def list_chats():
        if not os.path.exists(ui.save_folder):
            return {"chats": []}
        files = [f for f in os.listdir(ui.save_folder) if f.endswith(".json")]
        files.sort()
        return {"chats": files}

    @app.get("/chats/{name}")
    async def get_chat(name: str):
        file_path = os.path.join(ui.save_folder, name)
        if not os.path.exists(file_path):
            return []
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @app.get("/")
    async def index():
        with open(os.path.join(static_path, "app.html"), "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                data = await ws.receive_text()
                msg = json.loads(data)
                if msg.get("action") == "send_message":
                    text = msg.get("text", "")
                    queue: asyncio.Queue[str] = asyncio.Queue()

                    loop = asyncio.get_event_loop()

                    def run_stream():
                        for chunk, _c, _t in ui.stream(text, None):
                            asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
                        asyncio.run_coroutine_threadsafe(queue.put("__END__"), loop)
                        ui.save_messages()

                    future = loop.run_in_executor(None, run_stream)
                    while True:
                        token = await queue.get()
                        if token == "__END__":
                            await ws.send_text(json.dumps({"type": "done"}))
                            break
                        await ws.send_text(json.dumps({"type": "chunk", "data": token}))
                        await asyncio.sleep(0.03)
                    await future
                elif msg.get("action") == "edit_message":
                    idx = int(msg.get("index"))
                    ui.brain.update_message_content(msg.get("text", ""), idx)
                    ui.save_messages()
                elif msg.get("action") == "regenerate":
                    idx = int(msg.get("index"))
                    user_msg = ui.brain.messages[idx]["content"]
                    queue: asyncio.Queue[str] = asyncio.Queue()

                    loop = asyncio.get_event_loop()

                    def run_regen():
                        for chunk, _c, _t in ui.stream(user_msg, None):
                            asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
                        asyncio.run_coroutine_threadsafe(queue.put("__END__"), loop)
                        ui.save_messages()

                    future = loop.run_in_executor(None, run_regen)
                    while True:
                        token = await queue.get()
                        if token == "__END__":
                            await ws.send_text(json.dumps({"type": "done"}))
                            break
                        await ws.send_text(json.dumps({"type": "chunk", "data": token}))
                        await asyncio.sleep(0.03)
                    await future
                elif msg.get("action") == "new_chat":
                    ui.brain.set_messages([{"role": "system", "content": ui.system}])
                    ui.chat_filename = datetime.now().strftime(_DEFAULT_TIME_FORMAT)
                    ui.save_messages()
                elif msg.get("action") == "load_chat":
                    name = msg.get("name")
                    if name:
                        path = os.path.join(ui.save_folder, name)
                        if os.path.exists(path):
                            with open(path, "r", encoding="utf-8") as f:
                                ui.brain.set_messages(json.load(f))
                            ui.chat_filename = name.rsplit(".", 1)[0]
        except WebSocketDisconnect:
            pass

    return app
