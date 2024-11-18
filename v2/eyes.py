import struct
from typing import Generator
import websocket  # NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
import io
import random
import PIL.Image
from rich_pixels import Pixels
import bubble_painter
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
        self.client_id = str(uuid.uuid4())
        self.ws = websocket.WebSocket()

    def _connect(self):
        if not self.ws.connected:
            self.ws.connect(
                "ws://{}/ws?clientId={}".format(self.server_address, self.client_id)
            )

    def close(self):
        if self.ws.connected:
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
        clip_skip=None,
        dialog=False,
    ):

        lcm = lcm if lcm is not None else self.lcm
        negative = negative or self.default_negative_prompt
        dimensions = dimensions or self.default_dimensions
        checkpoint = checkpoint or self.default_checkpoint
        clip_skip = clip_skip or -1

        cl = CheckpointLoaderSimple(ckpt_name=checkpoint)
        l = EmptyLatentImage(width=dimensions[0], height=dimensions[1])
        cs = CLIPSetLastLayer(cl.outputs["CLIP"], clip_skip)
        pos = CLIPTextEncode(
            clip=cs.outputs["CLIP"],
            text=positive,
        )
        neg = CLIPTextEncode(clip=cs.outputs["CLIP"], text=negative)
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

        nodes = [cl, cs, l, pos, neg, ks, vaed]

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

        if dialog:
            return workflow(
                *nodes, SaveImageWebsocket(images=vaed.outputs["IMAGE"])
            ) | bubble_painter.clear_bubbles_workflow(vaed.outputs["IMAGE"])
        else:
            return workflow(
                *nodes,
                SaveImageWebsocket(images=vaed.outputs["IMAGE"]),
            )

    def queue_prompt(self, prompt):
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode("utf-8")
        req = urllib.request.Request(f"http://{self.server_address}/prompt", data=data)
        return json.loads(urllib.request.urlopen(req).read())

    def get_images(
        self, workflow
    ) -> Generator[dict[str, list[Image]] | None, None, None]:
        self._connect()
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
                    yield output_images

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
        clip_skip=None,
    ) -> tuple[Image, dict[str, list[Image]]] | tuple[None, None]:
        results = list(
            self.get_images(
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
                    cs=clip_skip,
                )
            )
        )[-1]
        if results is not None:
            keys = results.keys()
            key = tuple(filter(lambda key: "SaveImageWebsocket" in key, keys))[-1]
            return results[key][-1], results
        return None, None

    def generate_yield(
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
        clip_skip=None,
        dialog=None,
    ) -> Generator[tuple[Image | None, dict[str, list[Image]] | None], None, None]:
        for result in self.get_images(
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
                clip_skip=clip_skip,
                dialog=bool(dialog),
            )
        ):
            if result is not None:
                keys = result.keys()
                final_image = tuple(
                    filter(lambda key: "SaveImageWebsocket" in key, keys)
                )
                if (
                    final_image
                    and dialog
                    and len(
                        tuple(
                            filter(lambda key: "final" in key or "bubbles" in key, keys)
                        )
                    )
                    == 2
                ):
                    image = bubble_painter.add_text(result, dialog)
                    result[final_image[-1]][-1] = image
                yield result[final_image[-1]][-1] if final_image else None, result
            yield None, None

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
    gen = Eyes(default_checkpoint="waiANINSFWPONYXL_v80.safetensors")

    img, _ = gen.generate(
        "A portrait of Netta, a brunette Israeli 20 year old woman with green eyes.",
        steps=30,
        sampler_name="dpmpp_2m_sde_gpu",
    )
    if img is not None:
        img.show()
