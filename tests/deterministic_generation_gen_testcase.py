from dependency_modules.rewarding.app import MODEL as pipe
import torch
import os


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

os.makedirs("tests/images/", exist_ok=True)
prompt = "a portrait of a man with a beard"

for i in range(40):
    seed = i
    generator = torch.Generator().manual_seed(seed)
    result = pipe(prompt, generator=generator, num_inference_steps=40)
    images = result.images
    images[0].save(f"tests/images/{i}.webp")
