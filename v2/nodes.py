import base64
from dataclasses import dataclass, field
import io
from typing import Dict, Literal, Optional, Tuple, Union, TypeVar
import json
import itertools
from PIL.Image import Image
import numpy as np


T = TypeVar("T")
Output = Tuple[str, int]
Input = Union[T, Output]
Plug = Input[Output]
OUTPUT_ID = "_OUTPUT"


@dataclass
class Node:
    _output: bool = field(init=False, default=False)
    id: str = field(init=False, default="")
    _id_counter = itertools.count(0)
    _class_name: Optional[str] = field(init=False, default=None)
    _title: str = field(init=False, default="")
    _outputs: Tuple = field(init=False, default=())
    outputs: Dict[str, Input] = field(init=False)

    def __post_init__(self):
        self._class_name = self._class_name or self.__class__.__name__
        if not self._title:
            self._title = self._class_name

        self.id = f"{self._class_name}_{(next(Node._id_counter))}{OUTPUT_ID if self._output else ''}"

        self.outputs = {
            output: (str(self.id), i) for i, output in enumerate(self._outputs)
        }

    def rename(self, new_id: str):
        if self._output and not new_id.endswith(OUTPUT_ID):
            new_id += OUTPUT_ID
        self.id = new_id
        return self

    def json(self) -> Dict:
        return {
            self.id: {
                "inputs": {
                    key: self.__dict__[key]
                    for key in self.__dict__
                    if key not in ("id", "_outputs", "outputs")
                },
                "class_type": self._class_name,
                "_meta": {"title": self._title},
            }
        }

    @staticmethod
    def from_json(j: dict):

        inputs = [
            f"{key}: Plug" if isinstance(val, list) else f"{key}: Input = {repr(val)}"
            for key, val in j["inputs"].items()
        ]
        inputs = "\n\t".join(inputs)

        name = j["class_type"]
        title = j["_meta"]["title"]
        title = f'\t_title = "{title}"'

        c = f"""
@dataclass    
class {name}(Node):
\t{inputs}

{title}

 (\t_outputs = ()
    """
        return c


def workflow(*nodes: Node) -> str:
    j = {}
    for node in nodes:
        j.update(node.json())
    return j


@dataclass
class KSampler(Node):
    model: Plug
    positive: Plug
    negative: Plug
    latent_image: Plug

    seed: Input[int]
    steps: Input[int] = 20
    cfg: Input[float] = 8
    sampler_name: Input[str] = "dpmpp_sde"
    scheduler: Input[str] = "karras"
    denoise: Input[float] = 1

    _outputs = ("LATENT",)
    _output = True


@dataclass
class CheckpointLoaderSimple(Node):
    ckpt_name: Input[str] = "realisticVisionV51_v51VAE.safetensors"

    _title = "Load Checkpoint"

    _outputs = ("MODEL", "CLIP", "VAE")


@dataclass
class CLIPTextEncode(Node):
    clip: Plug
    text: Input[str]

    _title = "CLIP Text Encode (Prompt)"

    _outputs = ("CONDITIONING",)


@dataclass
class VAEDecode(Node):
    samples: Plug
    vae: Plug

    _title = "VAE Decode"

    _outputs = ("IMAGE",)


@dataclass
class EmptyLatentImage(Node):
    width: Input[int] = 512
    height: Input[int] = 816
    batch_size: Input = 1

    _title = "Empty Latent Image"

    _outputs = ("LATENT",)


@dataclass
class IPAdapterFaceID(Node):
    model: Plug
    ipadapter: Plug
    image: Plug
    clip_vision: Plug
    insightface: Plug
    weight: Input[float] = 0.85
    weight_faceidv2: Input[float] = 1
    weight_type: Input[str] = "linear"
    combine_embeds: Input[str] = "concat"
    start_at: Input[float] = 0
    end_at: Input[float] = 1
    embeds_scaling: Input[str] = "V only"

    _title = "IPAdapter FaceID"

    _outputs = ("MODEL", "face_image")


@dataclass
class IPAdapterModelLoader(Node):
    ipadapter_file: Input[str] = "ip-adapter-faceid-plusv2_sd15.bin"

    _title = "IPAdapter Model Loader"

    _outputs = ("IPADAPTER",)


@dataclass
class CLIPVisionLoader(Node):
    clip_name: Input[str] = "clip-vision_vit-h.safetensors"

    _title = "Load CLIP Vision"

    _outputs = ("CLIP_VISION",)


@dataclass
class IPAdapterInsightFaceLoader(Node):
    provider: Input[str] = "CPU"

    _title = "IPAdapter InsightFace Loader"

    _outputs = ("INSIGHTFACE",)


@dataclass
class ETN_LoadImageBase64(Node):
    image: Input[Union[str, Image]]

    _title = "Load Image (Base64)"

    _outputs = ("IMAGE", "MASK")

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.image, Image):
            img_byte_array = io.BytesIO()
            self.image.save(img_byte_array, format=self.image.format)
            img_byte_array = img_byte_array.getvalue()
            b64_encoded_bytes = base64.b64encode(img_byte_array)
            self.image = b64_encoded_bytes.decode("utf-8")


@dataclass
class ETN_SendImageWebSocket(Node):
    images: Plug
    format: Literal["PNG", "JPEG"] = "PNG"

    _title = "Send Image (WebSocket)"


@dataclass
class SaveImageWebsocket(Node):
    images: Plug
    _title = "SaveImageWebsocket"
    _output = True


@dataclass
class LoraLoader(Node):
    model: Plug
    clip: Plug
    lora_name: Input[str] = "lcm-lora-sdv1-5.safetensors"
    strength_model: Input[float] = 1
    strength_clip: Input[float] = 1

    _title = "Load LoRA"

    _outputs = ("MODEL", "CLIP")


@dataclass
class SamModelLoader(Node):
    model_name: Input[str] = "sam_vit_b (375MB)"
    _class_name = "SAMModelLoader (segment anything)"
    _title = "SAMModelLoader (segment anything)"

    _outputs = ("SAM_MODEL",)


@dataclass
class GroundingDinoModelLoader(Node):
    model_name: Input[str] = "GroundingDINO_SwinB (938MB)"
    _class_name = "GroundingDinoModelLoader (segment anything)"
    _title = "GroundingDinoModelLoader (segment anything)"

    _outputs = ("GROUNDING_DINO_MODEL",)


@dataclass
class GroundingDinoSAMSegment(Node):
    sam_model: Plug
    grounding_dino_model: Plug
    image: Input[np.ndarray]

    prompt: Input[str]
    threshold: Input[float] = 0.3

    _class_name = "GroundingDinoSAMSegment (segment anything)"
    _title = "GroundingDinoSAMSegment (segment anything)"

    _outputs = ("IMAGE", "MASK")


@dataclass
class InvertMask(Node):
    mask: Plug

    _outputs = ("MASK",)


@dataclass
class MaskToImage(Node):
    mask: Plug

    _title = "Convert Mask to Image"

    _outputs = ("IMAGE",)


@dataclass
class ImageCompositeMasked(Node):
    destination: Plug
    source: Plug
    mask: Plug

    x: Input[int] = 0
    y: Input[int] = 0
    resize_source: Input[bool] = False

    _outputs = ("IMAGE",)


if __name__ == "__main__":
    from rich import print

    print(
        workflow(
            (cl := CheckpointLoaderSimple()),
            (l := EmptyLatentImage()),
            (
                pos := CLIPTextEncode(
                    clip=cl.outputs["CLIP"],
                    text="20 year old mediterranean super model wearing a tight red bikini",
                )
            ),
            (
                neg := CLIPTextEncode(
                    clip=cl.outputs["CLIP"], text="artifacts, bad anatomy"
                )
            ),
            (
                ks := KSampler(
                    model=cl.outputs["MODEL"],
                    positive=pos.outputs["CONDITIONING"],
                    negative=neg.outputs["CONDITIONING"],
                    latent_image=l.outputs["LATENT"],
                    seed=1,
                )
            ),
            (vaed := VAEDecode(samples=ks.outputs["LATENT"], vae=cl.outputs["VAE"])),
            ETN_SendImageWebSocket(images=vaed.outputs["IMAGE"]),
        )
    )
