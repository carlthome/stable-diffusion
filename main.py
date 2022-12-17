import torch
from diffusers import StableDiffusionPipeline

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# TODO mps if macOS M1

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to(device)
pipe.enable_attention_slicing()

prompt = input("Text prompt: ")
image = pipe(prompt).images[0]

image.save(f"{prompt.lower().replace(' ', '_')}.png")
