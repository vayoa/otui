from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import json
from typing import Optional


def create_app(ui) -> FastAPI:
    app = FastAPI()

    app.mount("/", StaticFiles(directory="web/main page", html=True), name="static")

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

                    def run_stream():
                        for chunk, _content, _tool in ui.stream(text, None):
                            asyncio.run_coroutine_threadsafe(queue.put(chunk), asyncio.get_event_loop())
                        asyncio.run_coroutine_threadsafe(queue.put("__END__"), asyncio.get_event_loop())

                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, run_stream)
                    while True:
                        token = await queue.get()
                        if token == "__END__":
                            await ws.send_text(json.dumps({"type": "done"}))
                            break
                        await ws.send_text(json.dumps({"type": "chunk", "data": token}))
                        await asyncio.sleep(0.03)
                elif msg.get("action") == "edit_message":
                    idx = int(msg.get("index"))
                    ui.brain.update_message_content(msg.get("text", ""), idx)
                elif msg.get("action") == "regenerate":
                    idx = int(msg.get("index"))
                    user_msg = ui.brain.messages[idx]["content"]
                    queue: asyncio.Queue[str] = asyncio.Queue()

                    def run_regen():
                        for chunk, _c, _t in ui.stream(user_msg, None):
                            asyncio.run_coroutine_threadsafe(queue.put(chunk), asyncio.get_event_loop())
                        asyncio.run_coroutine_threadsafe(queue.put("__END__"), asyncio.get_event_loop())

                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, run_regen)
                    while True:
                        token = await queue.get()
                        if token == "__END__":
                            await ws.send_text(json.dumps({"type": "done"}))
                            break
                        await ws.send_text(json.dumps({"type": "chunk", "data": token}))
                        await asyncio.sleep(0.03)
                elif msg.get("action") == "new_chat":
                    ui.brain.set_messages([{"role": "system", "content": ui.system}])
        except WebSocketDisconnect:
            pass

    return app
