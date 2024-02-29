import torch
from generation_models.utils import (
    instantiate_from_config,
    pil_image_to_base64,
)
from typing import Dict, Any
import os

class ModelDeployment:
    def __init__(self, model_config):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        self.pipe = instantiate_from_config(model_config)

    async def generate(self, prompt_data: Dict[str, Any]):
        prompt_data = dict(prompt_data)
        generator = torch.manual_seed(prompt_data["seed"])
        image = self.pipe(
            generator=generator, **prompt_data, **prompt_data.get("pipeline_params", {})
        )
        base_64_image = pil_image_to_base64(image)
        return base_64_image
