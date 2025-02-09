import base64
from dataclasses import dataclass, field
import io
from typing import (
    Dict,
    ForwardRef,
    List,
    Literal,
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
    _json_inputs: dict = field(init=False)

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

        self._json_inputs = {
            key: self.__dict__[key]
            for key in self.__dict__
            if key not in ("id", "outputs", "_output_node", "_class_name", "_title")
        }

    def rename(self, new_id: str):
        if self._output_node and not new_id.endswith(OUTPUT_ID):
            new_id += OUTPUT_ID
        self.id = new_id
        return self

    def json(self) -> Dict:
        return {
            self.id: {
                "inputs": self._json_inputs,
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
BBOX_DETECTOR = NewType("BBOX_DETECTOR", _Out)
SEGM_DETECTOR = NewType("SEGM_DETECTOR", _Out)


@dataclass
class KSampler(Node[O("KSampler", {"LATENT": LATENT})]):
    model: MODEL
    positive: CONDITIONING
    negative: CONDITIONING
    latent_image: LATENT

    seed: int
    steps: int = 20
    cfg: float = 7
    sampler_name: str = "dpmpp_2m_sde_gpu"
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


@dataclass
class CLIPSetLastLayer(Node[O("CLIP Set Last Layer", {"CLIP": CLIP})]):
    clip: CLIP
    stop_at_clip_layer: int = -1


@dataclass
class UltralyticsDetectorProvider(
    Node[
        O(
            "UltralyticsDetectorProvider",
            {"BBOX_DETECTOR": BBOX_DETECTOR, "SEGM_DETECTOR": SEGM_DETECTOR},
        )
    ]
):
    model_name: str = "bbox/face_yolov8m.pt"


@dataclass
class SAMLoaderImpact(Node[O("SAMLoader (Impact)", {"SAM_MODEL": SAM_MODEL})]):
    model_name: str = "sam_vit_b_01ec64.pth"
    device_mode: Literal["AUTO", "Prefer GPU", "CPU"] = "AUTO"
    _class_name = "SAMLoader"


@dataclass
class FaceDetailer(Node[O("FaceDetailer", {"IMAGE": IMAGE})]):
    image: IMAGE
    model: MODEL
    clip: CLIP
    vae: VAE
    positive: CONDITIONING
    negative: CONDITIONING
    bbox_detector: BBOX_DETECTOR
    sam_model_opt: SAM_MODEL
    segm_detector_opt: SEGM_DETECTOR

    seed: int
    guide_size: int = 512
    guide_size_for: bool = True
    max_size: int = 1024
    steps: int = 10
    cfg: float = 7
    sampler_name: str = "dpmpp_2m_sde_gpu"
    scheduler: str = "karras"
    denoise: float = 0.5
    feather: int = 5
    noise_mask: bool = True
    force_inpaint: bool = True
    bbox_threshold: float = 0.5
    bbox_dilation: int = 10
    bbox_crop_factor: float = 3
    sam_detection_hint: str = "center-1"
    sam_dilation: int = 0
    sam_threshold: float = 0.93
    sam_bbox_expansion: int = 0
    sam_mask_hint_threshold: float = 0.7
    sam_mask_hint_use_negative: Literal["True", "False"] = "False"
    drop_size: int = 10
    wildcard: str = ""
    cycle: int = 1
    inpaint_model: bool = False
    noise_mask_feather: int = 20

    _output_node = True


@dataclass
class SolidMask(Node[O("SolidMask", {"MASK": MASK})]):
    value: float = 1.0
    width: int = 512
    height: int = 816


@dataclass
class MaskComposite(Node[O("MaskComposite", {"MASK": MASK})]):
    destination: MASK
    source: MASK
    x: int = 512
    y: int = 816
    operation: Literal["multiply", "add", "subtract", "or", "xor"] = "multiply"


@dataclass
class AttentionCoupleInput:
    conditioning: CONDITIONING
    mask: MASK


@dataclass
class AttentionCouple(Node[O("Attention Couple 🍌", {"MODEL": MODEL})]):
    model: MODEL
    base_mask: MASK
    inputs: list[AttentionCoupleInput]

    _class_name = "AttentionCouple|cgem156"

    def __post_init__(self):
        Node.__post_init__(self)
        self._json_inputs.pop("inputs")

        dynamic_inputs = [
            {f"cond_{i+1}": aci.conditioning, f"mask_{i+1}": aci.mask}
            for i, aci in enumerate(self.inputs)
        ]

        self._json_inputs = self._json_inputs | {
            k: v for di in dynamic_inputs for k, v in di.items()
        }


@dataclass
class ConditioningSetMask(
    Node[O("Conditioning (Set Mask)", {"CONDITIONING": CONDITIONING})]
):
    conditioning: CONDITIONING
    mask: MASK
    strength: float = 1.0
    set_cond_area: Literal["default", "mask bounds"] = "default"


@dataclass
class ImpactCombineConditionings(
    Node[O("Combine Conditionings", {"CONDITIONING": CONDITIONING})]
):
    conditionings: list[CONDITIONING]

    def __post_init__(self):
        Node.__post_init__(self)
        self._json_inputs.pop("conditionings")

        self._json_inputs = self._json_inputs | {
            f"conditioning{i+1}": conditioning
            for i, conditioning in enumerate(self.conditionings)
        }  # type: ignore


if __name__ == "__main__":
    from rich import print
    from eyes import Eyes

    # l = [AttentionCoupleInput(CONDITIONING(("1", 1)), MASK(("1", 1))) for i in range(4)]
    # ac = AttentionCouple(MODEL(("0", 12)), MASK(("0", 12)), l)
    # print(ac.json())

    seed = 4

    w = workflow(
        (cl := CheckpointLoaderSimple("ponyRealism_v22MainVAE.safetensors")),
        (l := EmptyLatentImage(width=1152, height=896)),
        (
            pos := CLIPTextEncode(
                clip=cl.outputs["CLIP"],
                text="score_9, score_8_up, score_7_up, An italian teenager with dark blonde hair is standing next to her mother, she is wearing a cheerleader outfit while her mother is wearing a belly dancer outfit, 2girls, the teenger is putting her hand around her mother's waist, looking at viewer, background is a family house, the teenager is saluting while the mother is squatting",
            )
        ),
        (
            pos1 := CLIPTextEncode(
                clip=cl.outputs["CLIP"],
                text="italian teenager with dark blonde hair wearing a red and white cheerleader outfit, looking at viewer, salute, serious, crying, straight on",
            )
        ),
        (
            pos2 := CLIPTextEncode(
                clip=cl.outputs["CLIP"],
                text="italian mother with bright blonde hair wearing a yellow belly dancer outfit, looking at viewer, chubby, ahegao, tongue out, huge smile, squatting",
            )
        ),
        (
            neg := CLIPTextEncode(
                clip=cl.outputs["CLIP"], text="score_6, score_5, score_4"
            )
        ),
        (smb := SolidMask(value=1.0, width=l.width, height=l.height)),
        (smfb := SolidMask(value=0.0, width=l.width, height=l.height)),
        (sm1 := SolidMask(value=1.0, width=(l.width // 2), height=l.height)),
        (
            mc1 := MaskComposite(
                destination=smfb.outputs["MASK"],
                source=sm1.outputs["MASK"],
                x=0,
                y=0,
                operation="add",
            )
        ),
        (
            mc2 := MaskComposite(
                destination=smfb.outputs["MASK"],
                source=sm1.outputs["MASK"],
                x=(l.width // 2),
                y=0,
                operation="add",
            )
        ),
        (
            attc := AttentionCouple(
                cl.outputs["MODEL"],
                smb.outputs["MASK"],
                inputs=[
                    AttentionCoupleInput(
                        conditioning=pos1.outputs["CONDITIONING"],
                        mask=mc1.outputs["MASK"],
                    ),
                    AttentionCoupleInput(
                        conditioning=pos2.outputs["CONDITIONING"],
                        mask=mc2.outputs["MASK"],
                    ),
                ],
            )
        ),
        (
            ks := KSampler(
                model=attc.outputs["MODEL"],
                positive=pos.outputs["CONDITIONING"],
                negative=neg.outputs["CONDITIONING"],
                latent_image=l.outputs["LATENT"],
                seed=seed,
            )
        ),
        (vaed := VAEDecode(samples=ks.outputs["LATENT"], vae=cl.outputs["VAE"])),
        SaveImageWebsocket(images=vaed.outputs["IMAGE"]),
        (
            csm1 := ConditioningSetMask(
                conditioning=pos1.outputs["CONDITIONING"],
                mask=mc1.outputs["MASK"],
                strength=0.85,
            )
        ),
        (
            csm2 := ConditioningSetMask(
                conditioning=pos2.outputs["CONDITIONING"],
                mask=mc2.outputs["MASK"],
                strength=0.85,
            )
        ),
        (
            combinecond := ImpactCombineConditionings(
                [
                    pos.outputs["CONDITIONING"],
                    csm1.outputs["CONDITIONING"],
                    csm2.outputs["CONDITIONING"],
                ]
            )
        ),
        (ult := UltralyticsDetectorProvider()),
        (samimp := SAMLoaderImpact()),
        (
            face := FaceDetailer(
                seed=seed,
                image=vaed.outputs["IMAGE"],
                model=cl.outputs["MODEL"],
                clip=cl.outputs["CLIP"],
                vae=cl.outputs["VAE"],
                positive=combinecond.outputs["CONDITIONING"],
                negative=neg.outputs["CONDITIONING"],
                bbox_detector=ult.outputs["BBOX_DETECTOR"],
                sam_model_opt=samimp.outputs["SAM_MODEL"],
                segm_detector_opt=ult.outputs["SEGM_DETECTOR"],
            )
        ),
        SaveImageWebsocket(images=face.outputs["IMAGE"]),
    )

    eyes = Eyes()
    v = None
    for r in eyes.get_images(w):
        if r is not None:
            v = r[list(r.keys())[-1]][-1]

    if v is not None:
        v.show()
