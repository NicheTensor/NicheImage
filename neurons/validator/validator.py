import time
import bittensor as bt
import random
import torch
from image_generation_subnet.protocol import ImageGenerating
from image_generation_subnet.base.validator import BaseValidatorNeuron
from neurons.validator.validator_proxy import ValidatorProxy
import image_generation_subnet as ig_subnet
import traceback
import asyncio


class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()
        self.supporting_models = {
            "RealisticVision": {
                "incentive_weight": 0.5,
                "checking_url": self.config.realistic_vision.check_url,
                "inference_params": {"num_inference_steps": 30},
                "timeout": 12,
            },
            "SDXLTurbo": {
                "incentive_weight": 0.5,
                "checking_url": self.config.sdxl_turbo.check_url,
                "inference_params": {"num_inference_steps": 4},
                "timeout": 4,
            },
        }

        self.update_active_models_func = ig_subnet.validator.update_active_models

        if self.config.proxy.port:
            try:
                self.validator_proxy = ValidatorProxy(self)
                bt.logging.info("Validator proxy started succesfully")
            except Exception as e:
                bt.logging.warning(
                    "Warning, proxy did not start correctly, so no one can query through your validator. Error message: "
                    + traceback.format_exc()
                )

        self.all_uids = [int(uid) for uid in self.metagraph.uids]
        self.all_uids_info = {
            str((uid.item())): {"scores": [], "model_name": "unknown"}
            for uid in self.metagraph.uids
        }

    def forward(self):
        """
        Validator forward pass. Consists of:
        - Querying all miners to get what model they run
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """

        self.update_active_models_func(self)

        for model_name in self.supporting_models.keys():
            seed = random.randint(0, 1e9)
            prompt = ig_subnet.validator.get_prompt(
                seed=seed, prompt_url=self.config.prompt_generating_endpoint
            )

            bt.logging.info(f"Received request for {model_name} model")
            bt.logging.info("Updating available models & uids")

            available_uids = [
                int(uid)
                for uid in self.all_uids_info.keys()
                if self.all_uids_info[uid]["model_name"] == model_name
            ]

            available_uids = self.filtering_miners(
                available_uids, model_name=model_name, seed=seed
            )

            if not available_uids:
                bt.logging.warning(
                    "No active miner available for specified model. Skipping setting weights."
                )
                continue
            else:
                bt.logging.info(f"Available uids: {available_uids}")

            synapse = ImageGenerating(
                prompt=prompt,
                seed=seed,
                model_name=model_name,
            )
            synapse.pipeline_params.update(
                self.supporting_models[model_name]["inference_params"]
            )
            responses = self.dendrite.query(
                axons=[self.metagraph.axons[uid] for uid in available_uids],
                synapse=synapse,
                deserialize=False,
            )

            bt.logging.info("Received responses, calculating rewards")
            checking_url = self.supporting_models[model_name]["checking_url"]
            rewards = ig_subnet.validator.get_reward(checking_url, responses, synapse)
            if rewards is None:
                return
            bt.logging.info(f"Scored responses: {rewards}")

            for i in range(len(available_uids)):
                self.all_uids_info[str(available_uids[i])]["scores"].append(rewards[i])
                self.all_uids_info[str(available_uids[i])][
                    "scores"
                ] = self.all_uids_info[str(available_uids[i])]["scores"][-10:]

        self.update_scores_on_chain()
        self.save_state()

    def filtering_miners(self, uids, model_name, seed):
        prompts = [
            ig_subnet.validator.get_prompt(
                seed=seed, prompt_url=self.config.prompt_generating_endpoint
            )
            for _ in range(len(uids))
        ]
        tasks = [
            asyncio.create_task(
                self.dendrite.forward(
                    [self.metagraph.axons[uid]],
                    ImageGenerating(prompt=prompt, seed=seed, model_name=model_name),
                    deserialize=False,
                    timeout=self.supporting_models[model_name]["timeout"],
                )
            )
            for uid, prompt in zip(uids, prompts)
        ]
        responses = asyncio.gather(*tasks)
        valid_uids = [uid for uid, response in zip(uids, responses) if response]
        return valid_uids

    def update_scores_on_chain(self):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        weights = torch.zeros(len(self.all_uids))

        for model_name in self.supporting_models.keys():
            model_specific_weights = torch.zeros(len(self.all_uids))

            for uid in self.all_uids_info.keys():
                if self.all_uids_info[uid]["model_name"] == model_name:
                    num_past_to_check = 5
                    model_specific_weights[int(uid)] = (
                        sum(self.all_uids_info[uid]["scores"][-num_past_to_check:])
                        / num_past_to_check
                    )

            tensor_sum = torch.sum(model_specific_weights)
            # Normalizing the tensor
            if tensor_sum > 0:
                model_specific_weights = model_specific_weights / tensor_sum
            else:
                continue
            # Correcting reward
            model_specific_weights = (
                model_specific_weights
                * self.supporting_models[model_name]["incentive_weight"]
            )
            bt.logging.info(f"model_specific_weights {model_specific_weights}")
            weights = weights + model_specific_weights

        bt.logging.info(f"weights {weights}")
        # Check if rewards contains NaN values.
        if torch.isnan(weights).any():
            bt.logging.warning(f"NaN values detected in weights: {weights}")
            # Replace any NaN values in rewards with 0.
            weights = torch.nan_to_num(weights, 0)

        bt.logging.debug(f"weights: {weights}")

        self.scores: torch.FloatTensor = weights
        bt.logging.info(f"Updated scores: {self.scores}")


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info("Validator running...", time.time())
            time.sleep(60)
