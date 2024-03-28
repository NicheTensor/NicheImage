import bittensor as bt
import pydantic
from generation_models.utils import base64_to_pil_image
import typing
import yaml
import traceback


MODEL_CONFIG = yaml.load(
    open("generation_models/configs/model_config.yaml"), yaml.FullLoader
)


class Information(bt.Synapse):
    request_dict: dict = {
        "get_miner_info": True,
    }
    response_dict: dict = {}


class ImageGenerating(bt.Synapse):
    prompt: str = pydantic.Field(
        default="",
        title="Prompt",
        description="Prompt for generation",
    )
    seed: int = pydantic.Field(
        default=0,
        title="Seed",
        description="Seed for generation",
    )
    model_name: str = pydantic.Field(
        default="",
        title="",
        description="Name of the model used for generation",
    )
    conditional_image: str = pydantic.Field(
        default="",
        title="Base64 Image",
        description="Base64 encoded image",
    )
    pipeline_type: str = pydantic.Field(
        default="txt2img",
        title="Pipeline Type",
        description="Type of pipeline used for generation, eg: txt2img, img2img, controlnet_txt2img",
    )
    pipeline_params: dict = pydantic.Field(
        default={},
        title="Pipeline Params",
        description="Dictionary of additional parameters for diffusers pipeline",
    )
    request_dict: dict = pydantic.Field(
        default={},
        title="Dictionary contains request",
        description="Dict contains arbitary information",
    )
    response_dict: dict = pydantic.Field(
        default={},
        title="Dictionary contains response",
        description="Dict contains arbitary information",
    )
    image = pydantic.Field(
        default="",
        title="Base64 Image",
        description="Base64 encoded image",
    )

    def miner_update(self, update: dict):
        return self.copy(update=update)

    def deserialize_input(self) -> dict:
        return self.deserialize()

    def limit_params(self):
        for k, v in self.pipeline_params.items():
            if k == "num_inference_steps":
                self.pipeline_params[k] = min(50, v)
        self.pipeline_params = self.pipeline_params

    def deserialize(self) -> dict:
        return {
            "prompt": self.prompt,
            "seed": self.seed,
            "model_name": self.model_name,
            "pipeline_type": self.pipeline_type,
            "pipeline_params": self.pipeline_params,
            "conditional_image": self.conditional_image,
            "image": self.image,
            "response_dict": self.response_dict,
        }

    def wandb_deserialize(self, uid) -> dict:
        import wandb

        image = base64_to_pil_image(self.image)
        prompt = self.prompt
        return {
            "images": {uid: wandb.Image(image, caption=prompt)},
        }


class TextGenerating(bt.Synapse):
    # Required request input, filled by sending dendrite caller.
    prompt_input: str = ""
    # Optional request output, filled by recieving axon.
    seed: int = 0
    request_dict: dict = {}
    model_name: str = ""
    prompt_output: typing.Optional[dict] = {}
    pipeline_params: dict = {}

    def miner_update(self, update: dict):
        self.prompt_output = update

    def deserialize_input(self) -> dict:
        deserialized_input = {
            "model": MODEL_CONFIG[self.model_name].get("repo_id", self.model_name),
            "prompt": [
                self.prompt_input,
            ],
        }
        deserialized_input.update(self.pipeline_params)
        return deserialized_input

    def deserialize(self) -> dict:
        """
        Deserialize the prompt output. This method retrieves the response from
        the miner in the form of prompt_output, deserializes it and returns it
        as the output of the dendrite.query() call.
        Returns:
        - dict: The deserialized response, which in this case is the value of prompt_output.
        """

        return {
            "prompt_output": self.prompt_output,
            "prompt_input": self.prompt_input,
            "model_name": self.model_name,
        }

    def wandb_deserialize(self, uid) -> dict:
        import wandb

        if self.prompt_output:
            try:
                data = [
                    [
                        self.prompt_input,
                        self.prompt_output["choices"][0]["text"],
                        self.model_name,
                    ]
                ]
                print(data)
                table = wandb.Table(
                    columns=["prompt_input", "prompt_output", "model_name"], data=data
                )
                return {
                    f"texts_{uid}": table,
                }
            except Exception as e:
                print(e)
                traceback.print_exc()
        else:
            bt.logging.info("No prompt output to log")
