import struct
import websocket  # NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
import io
import random
import base64
import PIL.Image
from rich_pixels import Pixels
from nodes import *


class Eyes:
    def __init__(
        self,
        default_checkpoint="realisticVisionV51_v51VAE.safetensors",
        server_address="127.0.0.1:8188",
        default_negative_prompt="text, watermark, artifacts, weird anatomy",
        default_dimensions=(512, 816),
        default_pixels_ratio=7,
        characters={},
        lcm=False,
    ):
        self.default_checkpoint = default_checkpoint
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
        positive,
        negative=None,
        character_image=None,
        dimensions=None,
        lcm=None,
        checkpoint=None,
        steps=None,
        sampler_name=None,
        cfg=None,
    ):

        lcm = lcm if lcm is not None else self.lcm
        negative = negative or self.default_negative_prompt
        dimensions = dimensions or self.default_dimensions
        checkpoint = checkpoint or self.default_checkpoint

        cl = CheckpointLoaderSimple(ckpt_name=checkpoint)
        l = EmptyLatentImage(width=dimensions[0], height=dimensions[1])
        pos = CLIPTextEncode(
            clip=cl.outputs["CLIP"],
            text=positive,
        )
        neg = CLIPTextEncode(clip=cl.outputs["CLIP"], text=negative)
        ks = KSampler(
            model=cl.outputs["MODEL"],
            positive=pos.outputs["CONDITIONING"],
            negative=neg.outputs["CONDITIONING"],
            latent_image=l.outputs["LATENT"],
            seed=random.randint(0, 10**10),
            steps=steps or 20,
            sampler_name=sampler_name or "dpmpp_sde",
            cfg=cfg or 8,
        )
        vaed = VAEDecode(samples=ks.outputs["LATENT"], vae=cl.outputs["VAE"])

        nodes = [cl, l, pos, neg, ks, vaed]

        if lcm:
            ll = LoraLoader(model=cl.outputs["MODEL"], clip=cl.outputs["CLIP"])

            pos.clip = ll.outputs["CLIP"]
            neg.clip = ll.outputs["CLIP"]
            ks.model = ll.outputs["MODEL"]
            ks.steps = 6
            ks.cfg = 2
            ks.sampler_name = "lcm"
            ks.scheduler = "sgm_uniform"

            nodes.append(ll)

        if character_image is not None:
            ipaml = IPAdapterModelLoader()
            cvl = CLIPVisionLoader()
            ipaifl = IPAdapterInsightFaceLoader()
            ci = ETN_LoadImageBase64(image=character_image)
            ipafi = IPAdapterFaceID(
                model=ll.outputs["MODEL"] if lcm else cl.outputs["MODEL"],
                ipadapter=ipaml.outputs["IPADAPTER"],
                image=ci.outputs["IMAGE"],
                clip_vision=cvl.outputs["CLIP_VISION"],
                insightface=ipaifl.outputs["INSIGHTFACE"],
            )

            ks.model = ipafi.outputs["MODEL"]

            nodes += [ipaml, cvl, ipaifl, ci, ipafi]

        return workflow(
            *nodes,
            SaveImageWebsocket(images=vaed.outputs["IMAGE"]),
        )

    def queue_prompt(self, prompt):
        self._connect()

        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode("utf-8")
        req = urllib.request.Request(f"http://{self.server_address}/prompt", data=data)
        return json.loads(urllib.request.urlopen(req).read())

    def get_images(self, workflow) -> dict[str, list[Image]] | None:
        prompt_id = self.queue_prompt(workflow)["prompt_id"]
        output_images = {}
        current_node = ""
        while True:
            out = self.ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message["type"] == "executing":
                    data = message["data"]
                    if data["prompt_id"] == prompt_id:
                        if data["node"] is None:
                            break  # Execution is done
                        else:
                            current_node = data["node"]
            else:
                if current_node.endswith(OUTPUT_ID):
                    images_output = output_images.get(current_node, [])
                    images_output.append(PIL.Image.open(io.BytesIO(out[8:])))
                    output_images[current_node] = images_output

        return output_images

    def generate(
        self,
        positive,
        negative=None,
        dimensions=None,
        character_image=None,
        lcm=None,
        checkpoint=None,
        steps=None,
        sampler_name=None,
        cfg=None,
    ) -> tuple[Image, dict[str, list[Image]]] | tuple[None, None]:
        results = self.get_images(
            self.get_workflow(
                positive,
                negative=negative,
                dimensions=dimensions,
                character_image=character_image,
                lcm=lcm,
                checkpoint=checkpoint,
                steps=steps,
                sampler_name=sampler_name,
                cfg=cfg,
            )
        )
        if results is not None:
            return results["SaveImageWebsocket"][-1], results
        return None, None

    def pixelize(self, img, ratio=None):
        if ratio is None:
            ratio = self.default_pixels_ratio

        return Pixels.from_image(
            img,
            resize=(int(img.width * ratio), int(img.height * ratio)),
        )

    def pixelize_save_show(self, img, img_dir, img_name=str(uuid.uuid4()), ratio=None):
        img_name = img_name.replace(" ", "+")
        img_dir = img_dir + f"\\{img_name}.png"
        img.save(img_dir)
        return f"[yellow link={img_dir}]show ({img_name})[/]", self.pixelize(
            img, ratio=ratio
        )


if __name__ == "__main__":
    from rich.console import Console, Group

    console = Console()

    gen = Eyes(default_checkpoint="waiANINSFWPONYXL_v80.safetensors")

    img = gen.generate(
        "A portrait of Netta, a brunette Israeli 20 year old woman with green eyes.",
        steps=30,
        sampler_name="dpmpp_2m_sde_gpu",
    )
    if img is not None:
        img.show()
        console.print(gen.pixelize(img))
