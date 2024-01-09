from fastapi import FastAPI, Request, Response, Depends
import bittensor as bt
import torch
from typing import List
from dependency_modules.rewarding.utils import instantiate_from_config, measure_time
from dependency_modules.rewarding.hash_compare import infer_hash
from dependency_modules.rewarding.similarity_compare import infer_similarity
from pydantic import BaseModel
import uvicorn
import argparse
import time
import threading
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
import yaml
import argparse
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

MODEL_CONFIG = yaml.load(open("configs/model_config.yaml"), yaml.FullLoader)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10002)
    parser.add_argument("--netuid", type=str, default=1)
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
        choices=list(MODEL_CONFIG.keys()),
        default="RealisticVision",
    )
    args = parser.parse_known_args()[0]
    return args


class Prompt(BaseModel):
    prompt: str
    seed: int
    images: List[str]
    model_name: str
    additional_params: dict = {}


app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

ARGS = get_args()
MODEL = instantiate_from_config(MODEL_CONFIG[ARGS.model_name])


@app.middleware("http")
@limiter.limit("60/minute")
async def filter_allowed_ips(request: Request, call_next):
    if ARGS.disable_secure:
        response = await call_next(request)
        return response
    if (request.client.host not in ALLOWED_IPS) and (
        request.client.host != "127.0.0.1"
    ):
        print(f"Blocking an unallowed ip:", request.client.host, flush=True)
        return Response(
            content="You do not have permission to access this resource",
            status_code=403,
        )
    print(f"Allow an ip:", request.client.host, flush=True)
    response = await call_next(request)
    return response


@app.post("/verify")
async def get_rewards(data: Prompt):
    generator = torch.manual_seed(data.seed)
    if len(data.images) == 0:
        return {"rewards": []}
    validator_result, pipe_time = measure_time(MODEL)(
        prompt=data.prompt, generator=generator, **data.additional_params
    )
    validator_image = validator_result.images[0]
    rewards, hash_time = measure_time(infer_similarity)(validator_image, data.images)
    return {
        "rewards": rewards,
        "component_time": {
            "pipe_time": pipe_time,
            "hash_time": hash_time,
        },
    }


def define_allowed_ips(url, netuid, min_stake):
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
    if not ARGS.disable_secure:
        allowed_ips_thread = threading.Thread(
            target=define_allowed_ips,
            args=(
                ARGS.chain_endpoint,
                ARGS.netuid,
                ARGS.min_stake,
            ),
        )
        allowed_ips_thread.setDaemon(True)
        allowed_ips_thread.start()
    uvicorn.run(app, host="0.0.0.0", port=ARGS.port)
