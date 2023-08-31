import argparse

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str)
parser.add_argument("--xl", action="store_true")
args = parser.parse_args()

if args.prompt:
    prompt = args.prompt
else:
    prompt = input("Text prompt: ")

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"{device=}")

if args.xl:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0"
    )
else:
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

pipe = pipe.to(device)
pipe.enable_attention_slicing()

image = pipe(prompt).images[0]

image.save(f"{prompt.lower().replace(' ', '_')}.png")
