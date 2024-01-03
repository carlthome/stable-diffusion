import argparse

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--xl", action="store_true")
    parser.add_argument("--turbo", action="store_true")
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
    elif args.turbo:
        if device == torch.device("cpu"):
            kwargs = {}
        else:
            kwargs = dict(torch_dtype=torch.float16, variant="fp16")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/sdxl-turbo", **kwargs
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    settings = {}

    if args.turbo:
        settings["num_inference_steps"] = 1
        settings["guidance_scale"] = 0.0

    image = pipe(prompt, **settings).images[0]

    image.save(f"{prompt.lower().replace(' ', '_')}.png")


if __name__ == "__main__":
    main()
