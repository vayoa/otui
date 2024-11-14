import base64
from dataclasses import dataclass, field
import io
from typing import (
    Dict,
    ForwardRef,
    List,
    NamedTuple,
    Optional,
    Tuple,
    TypedDict,
    Union,
    TypeVar,
    NewType,
    Type,
    Generic,
    Union,
    TypeVarTuple,
    get_origin,
    get_args,
    get_type_hints,
)
import json
import itertools
from PIL.Image import Image

T = TypeVar("T")
O = TypedDict
_Out = Tuple[str, int]
OUTPUT_ID = "_OUTPUT"

Model = NewType("Model", _Out)
Clip = NewType("Clip", _Out)


@dataclass
class Node(Generic[T]):
    _output_node: bool = field(init=False, default=False)
    id: str = field(init=False, default="")
    _id_counter = itertools.count(0)
    _class_name: Optional[str] = field(init=False, default=None)
    _title: str = field(init=False, default="")
    outputs: T = field(init=False)

    def __post_init__(self):
        self._class_name = self._class_name or self.__class__.__name__
        self.id = f"{self._class_name}_{(next(Node._id_counter))}{OUTPUT_ID if self._output_node else ''}"

        output_class = get_args(self.__orig_bases__[0])[0]  # type: ignore
        if isinstance(output_class, ForwardRef):
            output_class_name = output_class.__forward_arg__  # type: ignore
            self.outputs = OUTPUT_ID  # type: ignore
        else:
            output_class_name = output_class.__name__
            ts_types = output_class.__annotations__
            self.outputs = {t: ts_types[t]((self.id, i)) for i, t in enumerate(ts_types)}  # type: ignore

        if not self._title:
            self._title = output_class_name

    def rename(self, new_id: str):
        if self._output_node and not new_id.endswith(OUTPUT_ID):
            new_id += OUTPUT_ID
        self.id = new_id
        return self

    def json(self) -> Dict:
        return {
            self.id: {
                "inputs": {
                    key: self.__dict__[key]
                    for key in self.__dict__
                    if key
                    not in ("id", "outputs", "_output_node", "_class_name", "_title")
                },
                "class_type": self._class_name,
                "_meta": {"title": self._title},
            }
        }


def workflow(*nodes: Node) -> dict:
    j = {}
    for node in nodes:
        j.update(node.json())
    return j


MODEL = NewType("MODEL", _Out)
CONDITIONING = NewType("CONDITIONING", _Out)
LATENT = NewType("LATENT", _Out)
CLIP = NewType("CLIP", _Out)
VAE = NewType("VAE", _Out)
IMAGE = NewType("IMAGE", _Out)
IPADAPTER = NewType("IPADAPTER", _Out)
CLIP_VISION = NewType("CLIP_VISION", _Out)
INSIGHTFACE = NewType("INSIGHTFACE", _Out)
MASK = NewType("MASK", _Out)
SAM_MODEL = NewType("SAM_MODEL", _Out)
GROUNDING_DINO_MODEL = NewType("GROUNDING_DINO_MODEL", _Out)


@dataclass
class KSampler(Node[O("KSampler", {"LATENT": LATENT})]):
    model: MODEL
    positive: CONDITIONING
    negative: CONDITIONING
    latent_image: LATENT

    seed: int
    steps: int = 20
    cfg: float = 8
    sampler_name: str = "dpmpp_sde"
    scheduler: str = "karras"
    denoise: float = 1

    _output_node = True


@dataclass
class CheckpointLoaderSimple(
    Node[O("Load Checkpoint", {"MODEL": MODEL, "CLIP": CLIP, "VAE": VAE})]
):
    ckpt_name: str = "realisticVisionV51_v51VAE.safetensors"


@dataclass
class CLIPTextEncode(
    Node[O("CLIP Text Encode (Prompt)", {"CONDITIONING": CONDITIONING})]
):
    clip: CLIP
    text: str


@dataclass
class VAEDecode(Node[O("VAE Decode", {"IMAGE": IMAGE})]):
    samples: LATENT
    vae: VAE


@dataclass
class EmptyLatentImage(Node[O("Empty Latent Image", {"LATENT": LATENT})]):
    width: int = 512
    height: int = 816
    batch_size: int = 1


@dataclass
class IPAdapterFaceID(
    Node[O("IPAdapter FaceID", {"MODEL": MODEL, "face_image": IMAGE})]
):
    model: MODEL
    ipadapter: IPADAPTER
    image: IMAGE
    clip_vision: CLIP_VISION
    insightface: INSIGHTFACE
    weight: float = 0.85
    weight_faceidv2: float = 1
    weight_type: str = "linear"
    combine_embeds: str = "concat"
    start_at: float = 0
    end_at: float = 1
    embeds_scaling: str = "V only"


@dataclass
class IPAdapterModelLoader(Node[O("IPAdapter Model Loader", {"IPADAPTER": IPADAPTER})]):
    ipadapter_file: str = "ip-adapter-faceid-plusv2_sd15.bin"


@dataclass
class CLIPVisionLoader(Node[O("Load CLIP Vision", {"CLIP_VISION": CLIP_VISION})]):
    clip_name: str = "clip-vision_vit-h.safetensors"


@dataclass
class IPAdapterInsightFaceLoader(
    Node[O("IPAdapter InsightFace Loader", {"INSIGHTFACE": INSIGHTFACE})]
):
    provider: str = "CPU"


@dataclass
class ETN_LoadImageBase64(
    Node[O("Load Image (Base64)", {"IMAGE": IMAGE, "MASK": MASK})]
):
    image: Union[str, Image]

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.image, Image):
            img_byte_array = io.BytesIO()
            self.image.save(img_byte_array, format=self.image.format)
            img_byte_array = img_byte_array.getvalue()
            b64_encoded_bytes = base64.b64encode(img_byte_array)
            self.image = b64_encoded_bytes.decode("utf-8")


@dataclass
class SaveImageWebsocket(Node["SaveImageWebsocket"]):
    images: IMAGE
    _output_node = True


@dataclass
class LoraLoader(Node[O("Load LoRA", {"MODEL": MODEL, "CLIP": CLIP})]):
    model: MODEL
    clip: CLIP
    lora_name: str = "lcm-lora-sdv1-5.safetensors"
    strength_model: float = 1
    strength_clip: float = 1


@dataclass
class SamModelLoader(
    Node[O("SAMModelLoader (segment anything)", {"SAM_MODEL": SAM_MODEL})]
):
    model_name: str = "sam_vit_b (375MB)"
    _class_name = "SAMModelLoader (segment anything)"


@dataclass
class GroundingDinoModelLoader(
    Node[
        O(
            "SAMModelLoader (segment anything)",
            {"GROUNDING_DINO_MODEL": GROUNDING_DINO_MODEL},
        )
    ]
):
    model_name: str = "GroundingDINO_SwinB (938MB)"
    _class_name = "GroundingDinoModelLoader (segment anything)"


@dataclass
class GroundingDinoSAMSegment(
    Node[
        O("GroundingDinoSAMSegment (segment anything)", {"IMAGE": IMAGE, "MASK": MASK})
    ]
):
    sam_model: SAM_MODEL
    grounding_dino_model: GROUNDING_DINO_MODEL
    image: IMAGE

    prompt: str
    threshold: float = 0.3

    _class_name = "GroundingDinoSAMSegment (segment anything)"


@dataclass
class InvertMask(Node[O("InvertMask", {"MASK": MASK})]):
    mask: MASK


@dataclass
class MaskToImage(Node[O("Convert Mask to Image", {"IMAGE": IMAGE})]):
    mask: MASK


@dataclass
class ImageCompositeMasked(Node[O("ImageCompositeMasked", {"IMAGE": IMAGE})]):
    destination: IMAGE
    source: IMAGE
    mask: MASK

    x: int = 0
    y: int = 0
    resize_source: bool = False


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
            SaveImageWebsocket(images=vaed.outputs["IMAGE"]),
        )
    )
