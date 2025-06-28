from dia.model import Dia
import torch

model = Dia.from_local(
    r"E:\Documents\AIIG\config.json",
    r"E:\Documents\AIIG\dia-v0_1.pth",
    compute_dtype="float16",
    device=torch.device("cuda:0"),
)  # Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16")

text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."

print("Generating audio...")

output = model.generate(text, use_torch_compile=True, verbose=True)

print("Audio generation complete.")

model.save_audio("simple.mp3", output)

print("Audio saved as simple.mp3")
