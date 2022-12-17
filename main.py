import torch
from diffusers import StableDiffusionPipeline

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"{device=}")

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to(device)
pipe.enable_attention_slicing()

prompt = input("Text prompt: ")
image = pipe(prompt).images[0]

image.save(f"{prompt.lower().replace(' ', '_')}.png")
