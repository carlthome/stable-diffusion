import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("mps")
pipe.enable_attention_slicing()

prompt = input("Text prompt: ")
image = pipe(prompt).images[0]

image.save(f"{prompt.lower().replace(' ', '_')}.png")
