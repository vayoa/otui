# This is an example that uses the websockets api to know when a prompt execution is done
# Once the prompt execution is done it downloads the images using the /history endpoint

import struct
import websocket  # NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
import io
import random
import base64
from PIL import Image
from rich_pixels import Pixels
import os
import signal
import subprocess


class Generator:
    DEFAULT_SERVER_ADDRESS = "127.0.0.1:8188"
    DEFAULT_NEGATIVE_PROMPT = (
        "text, watermark, artifacts, weird anatomy, NSFW, naked, porn"
    )
    DEFAULT_DIMENSIONS = (512, 816)
    DEFAULT_PIXEL_RATIO = 7

    def __init__(
        self,
        server_address=DEFAULT_SERVER_ADDRESS,
        default_negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        default_dimensions=DEFAULT_DIMENSIONS,
        default_pixels_ratio=DEFAULT_PIXEL_RATIO,
        characters={},
        lcm=False,
    ):
        self.server_address = server_address
        self.default_negative_prompt = default_negative_prompt
        self.default_dimensions = default_dimensions
        self.default_pixels_ratio = default_pixels_ratio
        self.characters = characters
        self.lcm = lcm

        self.connected = False

    def _connect(self):
        if not self.connected:
            self.client_id = str(uuid.uuid4())
            self.ws = websocket.WebSocket()
            self.ws.connect(
                "ws://{}/ws?clientId={}".format(self.server_address, self.client_id)
            )

    def close(self):
        if self.connected:
            self.ws.close()

    def add_character(self, name, img, prompt):
        self.characters[name] = {"img": img, "prompt": prompt}

    def get_workflow(
        self,
        json_file,
        positive_prompt,
        negative_prompt=None,
        dimensions=None,
        lcm=None,
    ):
        if lcm is None:
            lcm = self.lcm

        if lcm:
            json_file = json_file.replace(".json", "_lcm.json")

        with open(json_file, "r") as file:
            workflow = json.load(file)

        if negative_prompt is None:
            negative_prompt = self.default_negative_prompt

        if dimensions is None:
            dimensions = self.default_dimensions

        workflow["3"]["inputs"]["seed"] = random.randint(0, 10**10)
        workflow["6"]["inputs"]["text"] = positive_prompt
        workflow["7"]["inputs"]["text"] = negative_prompt
        workflow["10"]["inputs"]["width"] = dimensions[0]
        workflow["10"]["inputs"]["height"] = dimensions[1]
        return workflow

    def queue_prompt(self, prompt):
        self._connect()

        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode("utf-8")
        req = urllib.request.Request(f"http://{self.server_address}/prompt", data=data)
        return json.loads(urllib.request.urlopen(req).read())

    def get_image(self, workflow):
        prompt_id = self.queue_prompt(workflow)["prompt_id"]
        while True:
            out = self.ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
            elif isinstance(out, bytes):
                s = struct.calcsize(">II")
                data = memoryview(out)
                if len(data) > s:
                    event, format = struct.unpack_from(">II", data)
                    # 1=PREVIEW_IMAGE, 2=PNG, see ComfyUI server.py
                    if event == 1 and format == 2:
                        return Image.open(io.BytesIO(data[s:]))
                break
            else:
                break
        return

    def generate(self, positive, negative=None, dimensions=None):
        workflow = self.get_workflow(
            "workflows/character_generator.json",
            positive,
            negative_prompt=negative,
            dimensions=dimensions,
        )
        return self.get_image(workflow)

    def generate_with_character(
        self,
        positive,
        character_image,
        negative=None,
        dimensions=None,
    ):
        workflow = self.get_workflow(
            "workflows/generator.json",
            positive,
            negative_prompt=negative,
            dimensions=dimensions,
        )

        img_byte_array = io.BytesIO()
        character_image.save(img_byte_array, format=character_image.format)
        img_byte_array = img_byte_array.getvalue()
        b64_encoded_bytes = base64.b64encode(img_byte_array)

        workflow["39"]["inputs"]["image"] = b64_encoded_bytes.decode("utf-8")

        return self.get_image(workflow)

    def pixelize(self, img, ratio=None):
        if ratio is None:
            ratio = self.default_pixels_ratio

        return Pixels.from_image(
            img,
            resize=(img.width // ratio, img.height // ratio),
        )

    def pixelize_save_show(self, img, img_dir, img_name=str(uuid.uuid4()), ratio=None):
        img_name = img_name.replace(" ", "+")
        img_dir = img_dir + f"\\{img_name}.png"
        img.save(img_dir)
        return f"[yellow link={img_dir}]show[/]", self.pixelize(img, ratio=ratio)


if __name__ == "__main__":
    from rich.console import Console, Group

    console = Console()

    gen = Generator(lcm=True)

    img = gen.generate(
        "A portrait of Netta, a brunette Israeli 20 year old woman with green eyes.",
    )
    img.show()
    console.print(gen.pixelize(img))
