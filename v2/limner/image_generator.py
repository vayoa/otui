import json
import base64
import requests
from eyes import Eyes
from limner import ui


def submit_post(url: str, data: dict):
    """
    Submit a POST request to the given URL with the given data.
    """
    return requests.post(url, data=json.dumps(data))


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
    image = ui.EYES.generate(
        positive=prompt,
        negative=negative_prompt,
        dimensions=(width, height),
        sampler_name=sampler_name,
        steps=steps,
        cfg=cfg_scale,
    )
    return image
