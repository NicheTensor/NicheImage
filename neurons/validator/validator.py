import time
import os
import bittensor as bt
import random
import torch
from image_generation_subnet.base.validator import BaseValidatorNeuron
from neurons.validator.validator_proxy import ValidatorProxy
from image_generation_subnet.validator import MinerManager
import image_generation_subnet as ig_subnet
import traceback
import yaml
import threading
import math
from copy import deepcopy
from image_generation_subnet.validator.offline_challenge import (
    get_backup_image,
    get_backup_prompt,
    get_backup_llm_prompt,
)

MODEL_CONFIGS = yaml.load(
    open("generation_models/configs/model_config.yaml"), yaml.FullLoader
)


class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.challenge_urls = {
            "txt2img": {
                "main": [self.config.challenge.prompt],
                "backup": [get_backup_prompt],
            },
            "img2img": {
                "main": [self.config.challenge.prompt, self.config.challenge.image],
                "backup": [get_backup_prompt, get_backup_image],
            },
            "controlnet_txt2img": {
                "main": [
                    self.config.challenge.prompt,
                    self.config.challenge.image,
                ],
                "backup": [get_backup_prompt, get_backup_image],
            },
            "gojourney": {
                "main": [
                    self.config.challenge.prompt,
                    ig_subnet.validator.get_promptGoJouney,
                ],
                "backup": [get_backup_prompt, ig_subnet.validator.get_promptGoJouney],
            },
            "text_generation": {
                "main": [self.config.challenge.llm_prompt],
                "backup": [get_backup_llm_prompt],
            },
        }
        # TODO: Balancing Incentive Weights
        self.nicheimage_catalogue = {
            "GoJourney": {
                "model_incentive_weight": 0.04,
                "supporting_pipelines": MODEL_CONFIGS["GoJourney"]["params"][
                    "supporting_pipelines"
                ],
                "reward_url": ig_subnet.validator.get_reward_GoJourney,
                "timeout": 12,
                "inference_params": {},
                "synapse_type": ig_subnet.protocol.ImageGenerating,
            },
            "DreamShaper": {
                "model_incentive_weight": 0.06,
                "supporting_pipelines": MODEL_CONFIGS["DreamShaper"]["params"][
                    "supporting_pipelines"
                ],
                "reward_url": self.config.reward_url.DreamShaper,
                "inference_params": {
                    "num_inference_steps": 30,
                    "width": 512,
                    "height": 768,
                    "guidance_scale": 7,
                    "negative_prompt": "out of frame, nude, duplicate, watermark, signature, mutated, text, blurry, worst quality, low quality, artificial, texture artifacts, jpeg artifacts",
                },
                "timeout": 12,
                "synapse_type": ig_subnet.protocol.ImageGenerating,
            },
            "RealisticVision": {
                "supporting_pipelines": MODEL_CONFIGS["RealisticVision"]["params"][
                    "supporting_pipelines"
                ],
                "model_incentive_weight": 0.25,
                "reward_url": self.config.reward_url.RealisticVision,
                "inference_params": {
                    "num_inference_steps": 30,
                    "negative_prompt": "out of frame, nude, duplicate, watermark, signature, mutated, text, blurry, worst quality, low quality, artificial, texture artifacts, jpeg artifacts",
                },
                "timeout": 12,
                "synapse_type": ig_subnet.protocol.ImageGenerating,
            },
            "RealitiesEdgeXL": {
                "supporting_pipelines": MODEL_CONFIGS["RealitiesEdgeXL"]["params"][
                    "supporting_pipelines"
                ],
                "model_incentive_weight": 0.30,
                "reward_url": self.config.reward_url.RealitiesEdgeXL,
                "inference_params": {
                    "num_inference_steps": 7,
                    "width": 1024,
                    "height": 1024,
                    "guidance_scale": 5.5,
                },
                "timeout": 12,
                "synapse_type": ig_subnet.protocol.ImageGenerating,
            },
            "AnimeV3": {
                "supporting_pipelines": MODEL_CONFIGS["AnimeV3"]["params"][
                    "supporting_pipelines"
                ],
                "model_incentive_weight": 0.34,
                "reward_url": self.config.reward_url.AnimeV3,
                "inference_params": {
                    "num_inference_steps": 25,
                    "width": 1024,
                    "height": 1024,
                    "guidance_scale": 7.0,
                    "negative_prompt": "out of frame, nude, duplicate, watermark, signature, mutated, text, blurry, worst quality, low quality, artificial, texture artifacts, jpeg artifacts",
                },
                "timeout": 12,
                "synapse_type": ig_subnet.protocol.ImageGenerating,
            },
            "Gemma7b": {
                "supporting_pipelines": MODEL_CONFIGS["Gemma7b"]["params"][
                    "supporting_pipelines"
                ],
                "model_incentive_weight": 0.01,
                "timeout": 64,
                "synapse_type": ig_subnet.protocol.TextGenerating,
                "reward_url": self.config.reward_url.Gemma7b,
                "inference_params": {},
            },
        }
        self.max_validate_batch = 5
        self.miner_manager = MinerManager(self)
        self.load_state()
        self.miner_manager.update_miners_identity()
        self.flattened_uids = []
        self.should_reward_indexes = []
        if self.config.use_wandb:
            self.init_wandb()
        self.wandb_data = {
            "all_uids_info": self.miner_manager.all_uids_info,
            "scores": {},
        }
        try:
            self.validator_proxy = ValidatorProxy(self)
            bt.logging.info("Validator proxy started succesfully")
        except Exception:
            if self.config.proxy.port:
                bt.logging.warning(
                    "Warning, proxy did not start correctly, so no one can query through your validator. Error message: "
                    + traceback.format_exc()
                )
            else:
                bt.logging.warning("Share validator info to owner failed")

    def init_wandb(self):
        import wandb

        config = deepcopy(self.config)

        run_name = f"validator-{self.uid}-{ig_subnet.__version__}"
        config.hotkey = self.wallet.hotkey.ss58_address
        config.run_name = run_name
        config.version = ig_subnet.__version__
        config.type = "validator"

        # Initialize the wandb run for the single project
        run = wandb.init(
            name=run_name,
            project="nicheimage",
            entity="toilaluan",
            config=config,
            dir=config.full_path,
            reinit=True,
        )

        # Sign the run to ensure it's from the correct hotkey
        signature = self.wallet.hotkey.sign(run.id.encode()).hex()
        config.signature = signature
        wandb.config.update(config, allow_val_change=True)

        bt.logging.success(
            f"Started wandb run for project '{run.project}', run '{run.name}'"
        )

    def forward(self):
        """
        Validator forward pass. Consists of:
        - Querying all miners to get their model_name
        - Forwarding requests to miners
        - Calculating rewards based on responses
        - Updating scores based on rewards
        - Saving the state
        """

        bt.logging.info("Updating available models & uids")
        num_forward_thread_per_loop = self.config.num_forward_thread_per_loop
        self.miner_manager.update_miners_identity()
        self.should_reward_indexes = self.update_flattened_uids()

        loop_base_time = self.config.loop_base_time  # default is 600 seconds
        forward_batch_size = len(self.flattened_uids) // num_forward_thread_per_loop
        if forward_batch_size == 0:
            forward_batch_size = len(self.flattened_uids)
        sleep_per_batch = loop_base_time / num_forward_thread_per_loop * 0.9
        bt.logging.info(
            (
                f"Forwarding {len(self.flattened_uids)} uids\n"
                f"{num_forward_thread_per_loop} threads will be used\n"
                f"Each forward pass will forward {forward_batch_size} uids\n"
                f"Sleeping {sleep_per_batch} seconds per batch to ensure the loop time is around {loop_base_time} seconds\n"
            )
        )
        threads = []
        loop_start = time.time()
        while self.flattened_uids:
            batch_uids = self.flattened_uids[:forward_batch_size]
            batch_should_reward_indexes = self.should_reward_indexes[
                :forward_batch_size
            ]
            batch_model_names = [
                self.miner_manager.all_uids_info[uid]["model_name"]
                for uid in batch_uids
            ]
            pipeline_types = [
                random.choice(
                    self.nicheimage_catalogue[model_name]["supporting_pipelines"]
                )
                for model_name in batch_model_names
            ]
            bt.logging.info(
                f"Forwarding {len(batch_uids)} uids\n {batch_uids}\n {batch_should_reward_indexes}"
            )
            thread = threading.Thread(
                target=self.async_query_and_reward,
                args=(
                    batch_uids,
                    batch_model_names,
                    pipeline_types,
                    batch_should_reward_indexes,
                ),
            )
            threads.append(thread)
            thread.start()
            del self.flattened_uids[:forward_batch_size]
            del self.should_reward_indexes[:forward_batch_size]
            if self.flattened_uids:
                bt.logging.info(f"Sleeping {sleep_per_batch} seconds before next batch")
                time.sleep(sleep_per_batch)
        for thread in threads:
            thread.join()
        self.update_scores_on_chain()
        self.save_state()
        self.wandb_data["scores"] = {k: v for k, v in enumerate(self.scores)}
        if self.config.use_wandb:
            try:
                import wandb

                wandb_uids_info = deepcopy(self.miner_manager.all_uids_info)
                for k, v in wandb_uids_info.items():
                    wandb_uids_info[k]["scores"] = (
                        sum(v["scores"]) / len(v["scores"]) if v["scores"] else 0
                    )
                self.wandb_data = {
                    "all_uids_info": wandb_uids_info,
                }
                wandb.log(self.wandb_data)
            except Exception:
                pass
        bt.logging.info(
            "Loop completed, uids info:\n",
            str(self.miner_manager.all_uids_info).replace("},", "},\n"),
        )

        actual_time_taken = time.time() - loop_start

        if actual_time_taken < loop_base_time:
            bt.logging.info(
                f"Sleeping for {loop_base_time - actual_time_taken} seconds"
            )
            time.sleep(loop_base_time - actual_time_taken)

    def update_flattened_uids(self):
        self.flattened_uids = []
        _uids = self.miner_manager.all_uids
        _model_names = [
            self.miner_manager.all_uids_info[uid]["model_name"] for uid in _uids
        ]
        rate_limit_per_uid = [
            self.miner_manager.all_uids_info[uid]["rate_limit"] for uid in _uids
        ]
        for uid, rate_limit, model_name in zip(_uids, rate_limit_per_uid, _model_names):
            rate_limit_usable = (
                1
                if self.config.debug_validator
                else int(math.ceil(rate_limit * self.config.volume_utilization_factor))
                - 1
            )
            if model_name in self.nicheimage_catalogue:
                self.flattened_uids = self.flattened_uids + [uid] * rate_limit_usable
        random.shuffle(self.flattened_uids)

        should_reward_indexes = [0] * len(self.flattened_uids)

        # Each uid should be rewarded only once
        uid_to_slots = {}
        for i, uid in enumerate(self.flattened_uids):
            uid_to_slots.setdefault(uid, []).append(i)
        for uid, slots in uid_to_slots.items():
            should_reward_indexes[random.choice(slots)] = 1
        return should_reward_indexes

    def async_query_and_reward(
        self,
        uids: list[int],
        model_names: list[str],
        pipeline_types: list[str],
        should_reward_indexes: list[int],
    ):
        dendrite = bt.dendrite(self.wallet)
        batch_by_model_pipeline = {}
        for uid, model_name, pipeline_type, should_reward in zip(
            uids, model_names, pipeline_types, should_reward_indexes
        ):
            batch_by_model_pipeline.setdefault((model_name, pipeline_type), []).append(
                (uid, should_reward)
            )
        for (
            model_name,
            pipeline_type,
        ), batched_uid_data in batch_by_model_pipeline.items():
            try:
                reward_url = self.nicheimage_catalogue[model_name]["reward_url"]
                synapses, batched_uid_data = self.prepare_challenge(
                    batched_uid_data, model_name, pipeline_type
                )
                for synapse, uid_data in zip(synapses, batched_uid_data):
                    bt.logging.info(f"Quering {uid_data}")
                    if not synapse:
                        continue
                    base_synapse = synapse.copy()
                    forward_uids = [uid for uid, _ in uid_data]
                    axons = [self.metagraph.axons[int(uid)] for uid, _ in uid_data]
                    should_rewards = [should_reward for _, should_reward in uid_data]
                    responses = dendrite.query(
                        axons=axons,
                        synapse=synapse,
                        deserialize=False,
                        timeout=self.nicheimage_catalogue[model_name]["timeout"],
                    )
                    if self.config.use_wandb:
                        for uid, response in zip(forward_uids, responses):
                            try:
                                import wandb

                                wandb_data = response.wandb_deserialize(uid)
                                wandb.log(wandb_data)
                            except Exception:
                                continue
                    reward_responses = [
                        response
                        for response, should_reward in zip(responses, should_rewards)
                        if should_reward
                    ]
                    reward_uids = [
                        uid for uid, should_reward in uid_data if should_reward
                    ]
                    bt.logging.info(
                        f"Received {len(responses)} responses, calculating rewards"
                    )
                    if reward_uids:
                        if callable(reward_url):
                            reward_uids, rewards = reward_url(
                                base_synapse, reward_responses, reward_uids
                            )
                        else:
                            reward_uids, rewards = ig_subnet.validator.get_reward(
                                reward_url,
                                base_synapse,
                                reward_responses,
                                reward_uids,
                                self.nicheimage_catalogue[model_name].get(
                                    "timeout", 12
                                ),
                            )

                        # Scale Reward based on Miner Volume
                        for i, uid in enumerate(reward_uids):
                            if rewards[i] > 0:
                                rewards[i] = rewards[i] * (
                                    0.9
                                    + 0.1
                                    * self.miner_manager.all_uids_info[uid][
                                        "reward_scale"
                                    ]
                                )

                        bt.logging.info(f"Scored responses: {rewards}")

                        self.miner_manager.update_scores(reward_uids, rewards)
            except Exception as e:
                bt.logging.error(
                    f"Error while processing forward pass for {model_name}: {e}"
                )
                bt.logging.error(traceback.print_exc())
                continue

    def prepare_challenge(self, available_uids, model_name, pipeline_type):
        synapse_type = self.nicheimage_catalogue[model_name]["synapse_type"]
        batch_size = random.randint(1, 5)
        random.shuffle(available_uids)
        batched_uids = [
            available_uids[i * batch_size : (i + 1) * batch_size]
            for i in range((len(available_uids) + batch_size - 1) // batch_size)
        ]
        num_batch = len(batched_uids)
        synapses = [
            synapse_type(pipeline_type=pipeline_type, model_name=model_name)
            for _ in range(num_batch)
        ]
        for synapse in synapses:
            synapse.pipeline_params.update(
                self.nicheimage_catalogue[model_name]["inference_params"]
            )
            synapse.seed = random.randint(0, 1e9)
        for challenge_url, backup_func in zip(
            self.challenge_urls[pipeline_type]["main"],
            self.challenge_urls[pipeline_type]["backup"],
        ):
            if callable(challenge_url):
                synapses = challenge_url(synapses)
            else:
                assert isinstance(challenge_url, str)
                synapses = ig_subnet.validator.get_challenge(
                    challenge_url, synapses, backup_func
                )
        return synapses, batched_uids

    def update_scores_on_chain(self):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        weights = torch.zeros(len(self.miner_manager.all_uids))
        for model_name in self.nicheimage_catalogue.keys():
            model_specific_weights = self.miner_manager.get_model_specific_weights(
                model_name
            )
            model_specific_weights = (
                model_specific_weights
                * self.nicheimage_catalogue[model_name]["model_incentive_weight"]
            )
            bt.logging.info(
                f"model_specific_weights for {model_name}\n{model_specific_weights}"
            )
            weights = weights + model_specific_weights

        # Check if rewards contains NaN values.
        if torch.isnan(weights).any():
            bt.logging.warning(f"NaN values detected in weights: {weights}")
            # Replace any NaN values in rewards with 0.
            weights = torch.nan_to_num(weights, 0)
        self.scores: torch.FloatTensor = weights
        bt.logging.success(f"Updated scores: {self.scores}")

    def save_state(self):
        """Saves the state of the validator to a file."""

        torch.save(
            {
                "step": self.step,
                "all_uids_info": self.miner_manager.all_uids_info,
            },
            self.config.neuron.full_path + "/state.pt",
        )

    def load_state(self):
        """Loads the state of the validator from a file."""

        # Load the state of the validator from file.
        try:
            path = self.config.neuron.full_path + "/state.pt"
            bt.logging.info("Loading validator state from: " + path)
            state = torch.load(path)
            self.step = state["step"]
            self.miner_manager.all_uids_info = state["all_uids_info"]
            bt.logging.info("Succesfully loaded state")
        except Exception as e:
            self.step = 0
            bt.logging.info("Could not find previously saved state.", e)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info("Validator running...", time.time())
            try:
                os.system("rm -rf wandb")
            except Exception:
                pass
            time.sleep(360)
