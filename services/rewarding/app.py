from fastapi import FastAPI, Request, Response
import bittensor as bt
from typing import List
from pydantic import BaseModel
import uvicorn
import argparse
import time
import threading
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
import yaml
from services.rays.image_generating import ModelDeployment
from ray import serve
from ray.serve.handle import DeploymentHandle
from services.rewarding.hash_compare import infer_hash
from discord_webhook import DiscordWebhook

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
        default="finney",
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
        "--probability_for_notice",
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


limiter = Limiter(key_func=get_remote_address)


class RewardApp:
    def __init__(self, model_handle: DeploymentHandle, args):
        self.model_handle = model_handle
        self.args = args
        self.app = FastAPI()
        self.app.add_api_route("/", self.__call__, methods=["POST"])
        self.app.middleware("http")(self.filter_allowed_ips)
        self.app.state.limiter = limiter
        self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        self.allowed_ips = []
        if args.webhook_url:
            self.webhook = DiscordWebhook(url=args.webhook_url, username=args.model_name)
        else:
            self.webhook = None

        if not self.args.disable_secure:
            self.allowed_ips_thread = threading.Thread(
                target=self.define_allowed_ips,
                args=(self.args.chain_endpoint, self.args.netuid, self.args.min_stake),
            )
            self.allowed_ips_thread.daemon = True
            self.allowed_ips_thread.start()

    async def __call__(self, reward_request: RewardRequest):
        base_data = reward_request.base_data
        miner_data = reward_request.miner_data
        validator_image = await self.model_handle.generate.remote(prompt_data=base_data)
        miner_images = [d.image for d in miner_data]
        rewards = infer_hash(validator_image, miner_images, self.webhook, self.args.probability_for_notice)
        rewards = [float(reward) for reward in rewards]
        return {
            "rewards": rewards,
        }

    @limiter.limit("120/minute")
    async def filter_allowed_ips(self, request: Request, call_next):
        if self.args.disable_secure:
            response = await call_next(request)
            return response
        if (request.client.host not in self.allowed_ips) and (
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
        while True:
            all_allowed_ips = []
            subtensor = bt.subtensor(url)
            metagraph = subtensor.metagraph(netuid)
            for uid in range(len(metagraph.total_stake)):
                if metagraph.total_stake[uid] > min_stake:
                    all_allowed_ips.append(metagraph.axons[uid].ip)
            self.allowed_ips = all_allowed_ips
            print("Updated allowed ips:", self.allowed_ips, flush=True)
            time.sleep(60)


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
