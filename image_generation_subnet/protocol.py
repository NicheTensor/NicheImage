import bittensor as bt
import pydantic
from generation_models.utils import base64_to_pil_image
import typing
import yaml
import requests
import traceback
import copy


MODEL_CONFIG = yaml.load(
    open("generation_models/configs/model_config.yaml"), yaml.FullLoader
)


class Information(bt.Synapse):
    request_dict: dict = {}
    response_dict: dict = {}
    computed_body_hash: str = pydantic.Field("", title="Computed Body Hash", frozen=False)

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
    image: str = pydantic.Field(
        default="",
        title="Base64 Image",
        description="Base64 encoded image",
    )
    # avoid warnings
    computed_body_hash: str = pydantic.Field("", title="Computed Body Hash", frozen=False)

    def miner_update(self, update: dict):
        return self.copy(update=update)

    def deserialize_input(self) -> dict:
        return self.deserialize()

    def limit_params(self):
        for k, v in self.pipeline_params.items():
            if k == "num_inference_steps":
                self.pipeline_params[k] = min(50, v)
            if k == "width":
                self.pipeline_params[k] = min(1536, v)
            if k == "height":
                self.pipeline_params[k] = min(1536, v)
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

    def deserialize_response(self):
        return {
            "image": self.image,
            "response_dict": self.response_dict,
        }

    def store_response(self, storage_url: str, uid, validator_uid):
        if self.model_name == "GoJourney":
            storage_url = storage_url + "/upload-go-journey-item"
            data = {
                "metadata": {
                    "miner_uid": uid,
                    "validator_uid": validator_uid,
                    "prompt": self.prompt,
                    "seed": self.seed,
                    "model_name": self.model_name,
                },
                "output": self.response_dict
            }
        else:
            storage_url = storage_url + "/upload-base64-item"
            data = {
                "image": self.image,
                "metadata": {
                    "miner_uid": uid,
                    "validator_uid": validator_uid,
                    "model_name": self.model_name,
                    "prompt": self.prompt,
                    "seed": self.seed,
                    "pipeline_type": self.pipeline_type,
                    "pipeline_params": self.pipeline_params,
                }
            }
        try:
            response = requests.post(storage_url, json=data)
            response.raise_for_status()
        except Exception as e:
            print(f"Error in storing response: {e}")
            traceback.print_exc()


class TextGenerating(bt.Synapse):
    # Required request input, filled by sending dendrite caller.
    prompt_input: str = ""
    # Optional request output, filled by recieving axon.
    seed: int = 0
    request_dict: dict = {}
    model_name: str = ""
    prompt_output: typing.Optional[dict] = {}
    pipeline_params: dict = {}
    # avoid warnings
    computed_body_hash: str = pydantic.Field("", title="Computed Body Hash", frozen=False)
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
    
    def limit_params(self):
        for k, v in self.pipeline_params.items():
            if k == "max_tokens":
                self.pipeline_params[k] = min(4096, v)
        self.pipeline_params = self.pipeline_params
        
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

    def deserialize_response(self):
        minimized_prompt_output: dict = copy.deepcopy(self.prompt_output)
        minimized_prompt_output['choices'][0].pop("logprobs")
        return {
            "prompt_output": minimized_prompt_output,
            "prompt_input": self.prompt_input,
            "model_name": self.model_name,
        }

    def store_response(self, storage_url: str, uid, validator_uid):
        storage_url = storage_url + "/upload-llm-item"
        minimized_prompt_output: dict = copy.deepcopy(self.prompt_output)
        minimized_prompt_output['choices'][0].pop("logprobs")
        data = {
            "prompt_input": self.prompt_input,
            "prompt_output": minimized_prompt_output,
            "metadata": {
                "miner_uid": uid,
                "validator_uid": validator_uid,
                "model": MODEL_CONFIG[self.model_name].get("repo_id", self.model_name),
                "model_name": self.model_name,
                "pipeline_params": self.pipeline_params,
            }
        }
        try:
            response = requests.post(storage_url, json=data)
            response.raise_for_status()
        except Exception as e:
            print(f"Error in storing response: {e}")
            traceback.print_exc()