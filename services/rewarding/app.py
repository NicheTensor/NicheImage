from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
import uvicorn
import argparse
import threading
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
import yaml
from services.rays.image_generating import ModelDeployment
from ray import serve
from ray.serve.handle import DeploymentHandle
from services.rewarding.cosine_similarity_compare import CosineSimilarityReward
from discord_webhook import AsyncDiscordWebhook
from services.rewarding.notice import notice_discord
import random
import asyncio
from generation_models.utils import base64_to_pil_image
from services.owner_api_core import define_allowed_ips, filter_allowed_ips, limiter

MODEL_CONFIG = yaml.load(
    open("generation_models/configs/model_config.yaml"), yaml.FullLoader
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--netuid", type=str, default=23)
    parser.add_argument("--min_stake", type=int, default=100)
    parser.add_argument(
        "--chain_endpoint",
        type=str,
        default="subtensor_fixed_imagenet.thinkiftechnology.com:9944",
    )
    parser.add_argument("--disable_secure", action="store_true")
    parser.add_argument(
        "--model_name",
        type=str,
        default="RealisticVision",
    )
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
    parser.add_argument(
        "--webhook_url",
        type=str,
        default="",
    )
    parser.add_argument(
        "--notice_prob",
        type=float,
        default=0.2,
    )

    args = parser.parse_args()
    return args


class Prompt(BaseModel):
    prompt: str
    seed: int
    image: str
    pipeline_type: str
    pipeline_params: dict = {}
    conditional_image: str = ""


class RewardRequest(BaseModel):
    miner_data: List[Prompt]
    base_data: Prompt


class RewardApp:
    def __init__(self, model_handle: DeploymentHandle, args):
        self.model_handle = model_handle
        self.args = args
        self.rewarder = CosineSimilarityReward()
        self.app = FastAPI()
        self.app.add_api_route("/", self.__call__, methods=["POST"])
        self.filter_allowed_ips = filter_allowed_ips
        self.app.middleware("http")(self.filter_allowed_ips)
        self.app.state.limiter = limiter
        self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        self.allowed_ips = []
        if args.webhook_url:
            self.webhook = AsyncDiscordWebhook(
                url=args.webhook_url, username=args.model_name
            )
        else:
            self.webhook = None

        if not self.args.disable_secure:
            self.allowed_ips_thread = threading.Thread(
                target=define_allowed_ips,
                args=(self.args.chain_endpoint, self.args.netuid, self.args.min_stake),
            )
            self.allowed_ips_thread.daemon = True
            self.allowed_ips_thread.start()

    async def __call__(self, reward_request: RewardRequest):
        base_data = reward_request.base_data
        miner_data = reward_request.miner_data
        validator_image = await self.model_handle.generate.remote(prompt_data=base_data)
        miner_images = [d.image for d in miner_data]
        rewards = self.rewarder.get_reward(validator_image, miner_images)
        rewards = [float(reward) for reward in rewards]
        print(rewards, flush=True)
        content = f"{str(rewards)}\n{str(dict(base_data))}"
        if self.webhook and random.random() < self.args.notice_prob:
            try:
                miner_images = [base64_to_pil_image(image) for image in miner_images]
                all_images = [base64_to_pil_image(validator_image)] + miner_images
                asyncio.create_task(notice_discord(all_images, self.webhook, content))
                print("Noticed discord")
            except Exception as e:
                print("Exception while noticing discord" + str(e), flush=True)
        return {
            "rewards": rewards,
        }


if __name__ == "__main__":
    args = get_args()
    model_deployment = serve.deployment(
        ModelDeployment,
        name="model_deployment",
        num_replicas=args.num_replicas,
        ray_actor_options={"num_gpus": args.num_gpus},
    )
    serve.run(
        model_deployment.bind(
            MODEL_CONFIG[args.model_name],
        ),
        name="model_deployment",
    )
    model_handle = serve.get_deployment_handle("model_deployment", "model_deployment")
    app = RewardApp(model_handle, args)
    uvicorn.run(
        app.app,
        host="0.0.0.0",
        port=args.port,
    )
