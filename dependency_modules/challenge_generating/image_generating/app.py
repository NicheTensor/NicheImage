import diffusers
import torch
from fastapi import FastAPI, Request, Response
import bittensor as bt
from pydantic import BaseModel, Extra
import uvicorn
import argparse
import time
import threading
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from generation_models.utils import pil_image_to_base64
from typing import Optional
from ray import serve
from ray.serve.handle import DeploymentHandle


class TextToImagePrompt(BaseModel, extra=Extra.allow):
    prompt: str
    negative_prompt: Optional[str] = "bad image, low quality, blurry"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--netuid", type=str, default=23)
    parser.add_argument("--min_stake", type=int, default=100)
    parser.add_argument(
        "--chain_endpoint",
        type=str,
        default="finney",
    )
    parser.add_argument("--disable_secure", action="store_true", default=False)
    parser.add_argument(
        "--num_gpus",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--num_replicas",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    return args


class ImageGenerator:
    def __init__(self):
        pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
        )
        pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )
        pipe.to("cuda")
        self.pipe = pipe

    async def __call__(self, data: dict):
        image = self.pipe(
            prompt=data["prompt"],
            negative_prompt=data["negative_prompt"],
            height=512,
            width=512,
            num_inference_steps=25,
        ).images[0]
        image = pil_image_to_base64(image)
        return image


limiter = Limiter(key_func=get_remote_address)


class ChallengeImage:
    def __init__(self, model_handle: DeploymentHandle, args):
        self.args = args
        self.model_handle = model_handle
        self.app = FastAPI()
        self.app.add_api_route("/", self.__call__, methods=["POST"])
        self.app.middleware("http")(self.filter_allowed_ips)
        self.app.state.limiter = limiter
        self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        if not self.args.disable_secure:
            self.allowed_ips_thread = threading.Thread(
                target=self.define_allowed_ips,
                args=(self.args.chain_endpoint, self.args.netuid, self.args.min_stake),
            )
            self.allowed_ips_thread.daemon = True
            self.allowed_ips_thread.start()

    async def __call__(
        self,
        data: TextToImagePrompt,
    ):
        data = dict(data)
        image = await self.model_handle.remote(data)
        return {"conditional_image": image}

    @limiter.limit("120/minute")
    async def filter_allowed_ips(self, request: Request, call_next):
        if self.args.disable_secure:
            response = await call_next(request)
            return response
        if (request.client.host not in ALLOWED_IPS) and (
            request.client.host != "127.0.0.1"
        ):
            print("Blocking an unallowed ip:", request.client.host, flush=True)
            return Response(
                content="You do not have permission to access this resource",
                status_code=403,
            )
        print("Allow an ip:", request.client.host, flush=True)
        response = await call_next(request)
        return response

    def define_allowed_ips(self, url, netuid, min_stake):
        global ALLOWED_IPS
        ALLOWED_IPS = []
        while True:
            all_allowed_ips = []
            subtensor = bt.subtensor(url)
            metagraph = subtensor.metagraph(netuid)
            for uid in range(len(metagraph.total_stake)):
                if metagraph.total_stake[uid] > min_stake:
                    all_allowed_ips.append(metagraph.axons[uid].ip)
            ALLOWED_IPS = all_allowed_ips
            print("Updated allowed ips:", ALLOWED_IPS, flush=True)
            time.sleep(60)


if __name__ == "__main__":
    args = get_args()
    print(args)
    model_deployment = serve.deployment(
        ImageGenerator,
        name="deployment",
        num_replicas=args.num_replicas,
        ray_actor_options={"num_gpus": args.num_gpus},
    )
    serve.run(
        model_deployment.bind(),
        name="deployment-image-challenge",
    )
    model_handle = serve.get_deployment_handle(
        "deployment", "deployment-image-challenge"
    )
    app = ChallengeImage(model_handle, args)
    uvicorn.run(app.app, host="0.0.0.0", port=args.port)
