from .base_model import BaseModel
import openai
import torch
import os
import random
from openai import OpenAI
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from image_generation_subnet.utils.moderation_model import Moderation


class NicheDallE(BaseModel):
    def __init__(self, *args, **kwargs):
        self.inference_function = self.load_model(*args, **kwargs)
        self.client = OpenAI()
        self.moderation = Moderation()

    def load_model(self, *args, **kwargs):
        imagine_inference_function = self.load_imagine(*args, **kwargs)
        return imagine_inference_function

    def __call__(self, *args, **kwargs):
        return self.inference_function(*args, **kwargs)

    def load_imagine(self, *args, **kwargs):
        supporting_sizes = ["1792x1024", "1024x1792"]

        def inference_function(*args, **kwargs):
            prompt = kwargs.get("prompt", "a cute cat")
            # check prompt safety
            flagged, response = self.moderation(prompt)
            if flagged:
                print(response)
                data = {
                    "url": "",
                    "revised_prompt": "",
                }
            else:
                style = kwargs.get("style", "natural")
                if style not in ["vivid", "natural"]:
                    style = "natural"
                size = kwargs.get("size", random.choice(supporting_sizes))
                if size not in supporting_sizes:
                    size = random.choice(supporting_sizes)
                response_obj = self.client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    n=1,
                    size=size,
                    response_format="url",
                    style=style,
                )
                data = {
                    "url": response_obj.data[0].url,
                    "revised_prompt": response_obj.data[0].revised_prompt,
                }
            print(data, flush=True)
            return data

        return inference_function
