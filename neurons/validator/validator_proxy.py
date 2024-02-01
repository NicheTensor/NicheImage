from fastapi import FastAPI, HTTPException, Depends
from concurrent.futures import ThreadPoolExecutor
import requests
import image_generation_subnet.protocol as protocol
import uvicorn
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature
import bittensor as bt
import base64
import image_generation_subnet
import os
import random
import asyncio
from image_generation_subnet.validator.proxy import ProxyCounter
import traceback


class ValidatorProxy:
    def __init__(
        self,
        validator,
    ):
        self.validator = validator
        self.verify_credentials = self.get_credentials()
        self.miner_request_counter = {}
        self.dendrite = bt.dendrite(wallet=validator.wallet)
        self.app = FastAPI()
        self.app.add_api_route(
            "/validator_proxy",
            self.forward,
            methods=["POST"],
            dependencies=[Depends(self.get_self)],
        )
        self.loop = asyncio.get_event_loop()
        self.proxy_counter = ProxyCounter(
            os.path.join(self.validator.config.neuron.full_path, "proxy_counter.json")
        )
        self.start_server()

    def get_credentials(self):
        response = requests.post(
            f"{self.validator.config.proxy.proxy_client_url}/get_credentials",
            json={
                "postfix": f":{self.validator.config.proxy.port}/validator_proxy",
                "uid": self.validator.uid,
            },
            timeout=30,
        )
        if response.status_code != 200:
            raise Exception("Error getting credentials from market api")
        response = response.json()
        message = response["message"]
        signature = response["signature"]
        signature = base64.b64decode(signature)

        def verify_credentials(public_key_bytes):
            public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
            try:
                public_key.verify(signature, message.encode("utf-8"))
            except InvalidSignature:
                raise Exception("Invalid signature")

        return verify_credentials

    def start_server(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.executor.submit(
            uvicorn.run, self.app, host="0.0.0.0", port=self.validator.config.proxy.port
        )

    def authenticate_token(self, public_key_bytes):
        public_key_bytes = base64.b64decode(public_key_bytes)
        try:
            self.verify_credentials(public_key_bytes)
            bt.logging.info("Successfully authenticated token")
            return public_key_bytes
        except Exception as e:
            print("Exception occured in authenticating token", e, flush=True)
            raise HTTPException(
                status_code=401, detail="Error getting authentication token"
            )

    def random_check(self, uid, synapse, url):
        if random.random() < self.validator.config.proxy.checking_probability:
            bt.logging.info(f"Random check for miner {uid}")
            rewards = image_generation_subnet.validator.get_reward(url, [synapse])
            if rewards is None:
                return False
            self.validator.all_uids_info[uid]["scores"].append(rewards[0])
            bt.logging.info(f"Proxy check responses: {rewards}")
            return rewards[0] > 0
        else:
            bt.logging.info("Not doing random check")
            return True

    async def forward(self, data: dict = {}):
        self.authenticate_token(data["authorization"])
        payload = data.get("payload")
        miner_uid = data.get("miner_uid", -1)
        if "recheck" in payload:
            bt.logging.info("Rechecking validators")
            self.verify_credentials = self.get_credentials()
            return {"message": "Rechecked"}
        try:
            bt.logging.info("Received a request!")
            if "seed" not in payload:
                payload["seed"] = random.randint(0, 1e9)
            model_name = payload["model_name"]
            category = payload["category"]
            synapse_cls = self.validator.category_models[category]["base_synapse"]
            synapse = synapse_cls(**payload)

            # Override default pipeline params
            for k, v in self.validator.category_models[synapse.category]["models"][
                synapse.model_name
            ]["inference_params"].items():
                if k not in synapse.pipeline_params:
                    synapse.pipeline_params[k] = v

            # Limit inference steps
            if "num_inference_steps" in synapse.pipeline_params:
                synapse.pipeline_params["num_inference_steps"] = min(
                    50, synapse.pipeline_params["num_inference_steps"]
                )
            timeout = 20
            scores = self.validator.scores
            metagraph = self.validator.metagraph
            reward_url = self.validator.category_models[category]["models"][model_name][
                "reward_url"
            ]

            if miner_uid >= 0:
                available_uids = [miner_uid]
                miner_indexes = [0]
            else:
                available_uids = [
                    int(uid)
                    for uid in self.validator.all_uids_info.keys()
                    if self.validator.all_uids_info[uid]["model_name"] == model_name
                    and self.validator.all_uids_info[uid]["category"] == category
                ]

                scores = [
                    self.validator.all_uids_info[uid]["scores"]
                    for uid in available_uids
                ]

                scores = [sum(s) / max(1, len(s)) for s in scores]
                bt.logging.info(f"Available uids: {available_uids}")

                good_uids_indexes = [
                    i
                    for i in range(len(available_uids))
                    if scores[i] > self.validator.config.proxy.miner_score_threshold
                ]

                if len(good_uids_indexes) == 0:
                    good_uids_indexes = [
                        i for i in range(len(available_uids)) if scores[i] > 0
                    ]
                if len(good_uids_indexes) == 0:
                    raise Exception("No miners meet the score threshold")

                available_uids = [available_uids[index] for index in good_uids_indexes]
                scores = [scores[index] for index in good_uids_indexes]

                miner_indexes = list(range(len(available_uids)))
            random.shuffle(miner_indexes)
            is_valid_response = False
            for miner_uid_index in miner_indexes:
                bt.logging.info(f"Selected miner index: {miner_uid_index}")
                miner_uid = available_uids[miner_uid_index]
                bt.logging.info(f"Selected miner uid: {miner_uid}")
                bt.logging.info(
                    f"Forwarding request to miner {miner_uid} with score {scores[miner_uid_index]}"
                )
                axon = metagraph.axons[miner_uid]
                bt.logging.info(f"Sending request to axon: {axon}")
                task = asyncio.create_task(
                    self.dendrite.forward(
                        [axon],
                        synapse,
                        deserialize=False,
                        timeout=timeout,
                    )
                )
                await asyncio.gather(task)
                response = task.result()[0]
                bt.logging.info(f"Received responses")
                if not len(response.image):
                    bt.logging.info("No image in response")
                    continue
                else:
                    bt.logging.info("Image in response")

                if self.random_check(
                    miner_uid,
                    synapse,
                    reward_url,
                ):
                    is_valid_response = True
                    bt.logging.info("Checked OK")
                    break
            if not is_valid_response:
                raise Exception("No valid response")
            try:
                self.proxy_counter.update(is_success=is_valid_response)
                self.proxy_counter.save()
            except Exception:
                print(
                    "Exception occured in updating proxy counter",
                    traceback.format_exc(),
                    flush=True,
                )
            return response.deserialize()
        except Exception as e:
            print(
                "Exception occured in proxy forward", traceback.format_exc(), flush=True
            )
            raise HTTPException(status_code=400, detail=str(e))

    async def get_self(self):
        return self
