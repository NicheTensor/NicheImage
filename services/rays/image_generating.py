import torch
from services.rewarding.utils import (
    instantiate_from_config,
    pil_image_to_base64,
)
from typing import Dict, Any


class ModelDeployment:
    def __init__(self, model_config):
        self.pipe = instantiate_from_config(model_config)

    async def generate(self, prompt_data: Dict[str, Any]):
        generator = torch.manual_seed(prompt_data["seed"])
        image = self.pipe(
            generator=generator, **prompt_data, **prompt_data.get("pipeline_params", {})
        ).images[0]
        base_64_image = pil_image_to_base64(image)
        return base_64_image
