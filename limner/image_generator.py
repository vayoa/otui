import json
import base64
import requests
from eyes import Eyes

EYES = Eyes(default_checkpoint="waiANINSFWPONYXL_v80.safetensors")


def generate(
    prompt,
    negative_prompt="",
    sampler_name="dpmpp_2m_sde_gpu",
    batch_size=1,
    steps=25,
    cfg_scale=7.0,
    width=512,
    height=512,
    controlnet_payload=None,
):
    return EYES.generate(
        positive=prompt,
        negative=negative_prompt,
        dimensions=(width, height),
        sampler_name=sampler_name,
        steps=steps,
        cfg=cfg_scale,
    )[0]
