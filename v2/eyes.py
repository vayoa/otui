import struct
from typing import Generator, TypedDict
import websocket  # NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
from urllib.error import URLError
import io
import random
import PIL.Image
from rich_pixels import Pixels
import bubble_painter
from nodes import *
import time


class Section(TypedDict):
    width: int
    height: int
    x: int
    y: int
    prompt: str


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
        while True:
            try:
                if self.ws is None or not self.ws.connected:
                    self.ws = websocket.WebSocket()
                    self.ws.connect(
                        "ws://{}/ws?clientId={}".format(
                            self.server_address, self.client_id
                        )
                    )
                break
            except Exception as e:
                print(f"Connection failed. Retrying in 5 seconds...\n{e}")
                time.sleep(5)

    def close(self):
        if self.ws and self.ws.connected:
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
        face_detailer=False,
        sections: list[Section] | None = None,
    ):

        nodes = []
        attcnodes = []

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

        if sections:
            smb = SolidMask(value=1.0, width=l.width, height=l.height)
            smfb = SolidMask(value=0.0, width=l.width, height=l.height)
            nodes += [smb, smfb]

            for section in sections:
                spos = CLIPTextEncode(clip=cl.outputs["CLIP"], text=section["prompt"])
                sm = SolidMask(
                    value=1.0, width=section["width"], height=section["height"]
                )
                mc = MaskComposite(
                    destination=smfb.outputs["MASK"],
                    source=sm.outputs["MASK"],
                    x=section["x"],
                    y=section["y"],
                    operation="add",
                )
                part = [spos, sm, mc]
                attcnodes.append(part)
                nodes += part

            attc = AttentionCouple(
                cl.outputs["MODEL"],
                smb.outputs["MASK"],
                inputs=[
                    AttentionCoupleInput(
                        conditioning=spos.outputs["CONDITIONING"],
                        mask=mc.outputs["MASK"],
                    )
                    for spos, sm, mc in attcnodes
                ],
            )
            nodes.append(attc)

        neg = CLIPTextEncode(clip=cs.outputs["CLIP"], text=negative)
        ks = KSampler(
            model=attc.outputs["MODEL"] if sections else cl.outputs["MODEL"],
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

        output_image = vaed.outputs["IMAGE"]

        if face_detailer:
            if sections:
                section_conditions = []
                for spos, sm, mc in attcnodes:
                    csm = ConditioningSetMask(
                        conditioning=spos.outputs["CONDITIONING"],
                        mask=mc.outputs["MASK"],
                        strength=0.85,
                    )
                    section_conditions.append(csm)
                    nodes.append(csm)

                combinecond = ImpactCombineConditionings(
                    [pos.outputs["CONDITIONING"], *section_conditions]
                )
                nodes.append(combinecond)

            ult = UltralyticsDetectorProvider()
            samimp = SAMLoaderImpact()
            size = dimensions[0] * dimensions[1]
            face = FaceDetailer(
                seed=random.randint(0, 10**10),
                model=cl.outputs["MODEL"],
                image=vaed.outputs["IMAGE"],
                clip=cl.outputs["CLIP"],
                vae=cl.outputs["VAE"],
                positive=(
                    combinecond.outputs["CONDITIONING"]
                    if sections
                    else pos.outputs["CONDITIONING"]
                ),
                negative=neg.outputs["CONDITIONING"],
                bbox_detector=ult.outputs["BBOX_DETECTOR"],
                sam_model_opt=samimp.outputs["SAM_MODEL"],
                segm_detector_opt=ult.outputs["SEGM_DETECTOR"],
                steps=8 if size <= 1_032_192 else 10,
            )
            nodes += [ult, samimp, face]
            output_image = face.outputs["IMAGE"]

        if dialog:
            return workflow(
                *nodes, SaveImageWebsocket(images=output_image)
            ) | bubble_painter.clear_bubbles_workflow(output_image)
        else:
            return workflow(
                *nodes,
                SaveImageWebsocket(images=output_image),
            )

    def queue_prompt(self, prompt):
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode("utf-8")
        req = urllib.request.Request(f"http://{self.server_address}/prompt", data=data)
        while True:  # Keep retrying until the server is back online
            try:
                return json.loads(urllib.request.urlopen(req).read())
            except URLError as e:
                print(f"Connection error: {e}. Retrying in 5 seconds...")
                time.sleep(5)  # Wait for 5 seconds before retrying

    def get_images(
        self, workflow
    ) -> Generator[dict[str, list[Image]] | None, None, None]:
        self._connect()
        prompt_id = self.queue_prompt(workflow)["prompt_id"]
        output_images = {}
        current_node = ""
        while True:
            try:
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
            except (websocket.WebSocketException, ConnectionResetError):
                self._connect()

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
                    clip_skip=clip_skip,
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
        face_detailer=None,
        sections=None,
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
                face_detailer=bool(face_detailer),
                sections=sections,
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

    def interrupt(self):
        while True:  # Keep retrying until the server is back online
            try:
                urllib.request.urlopen(
                    f"http://{self.server_address}/interrupt", data=b"{}"
                )
            except URLError as e:
                print(f"Connection error: {e}. Retrying in 5 seconds...")
                time.sleep(5)  # Wait for 5 seconds before retrying


if __name__ == "__main__":
    gen = Eyes(default_checkpoint="waiANINSFWPONYXL_v80.safetensors")

    img, _ = gen.generate(
        "A portrait of Netta, a brunette Israeli 20 year old woman with green eyes.",
        steps=30,
        sampler_name="dpmpp_2m_sde_gpu",
    )
    if img is not None:
        img.show()
