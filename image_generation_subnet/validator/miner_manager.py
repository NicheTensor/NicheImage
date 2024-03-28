import bittensor as bt
from image_generation_subnet.protocol import ImageGenerating
import torch
from image_generation_subnet.utils.volume_setting import get_volume_per_validator


class MinerManager:
    def __init__(self, validator):
        self.validator = validator
        self.all_uids = [int(uid.item()) for uid in self.validator.metagraph.uids]
        self.all_uids_info = {
            uid: {"scores": [], "model_name": ""} for uid in self.all_uids
        }

    def get_miner_info(self):
        """
        1. Query model_name of available uids
        """
        self.all_uids = [int(uid) for uid in self.validator.metagraph.uids]
        uid_to_axon = dict(zip(self.all_uids, self.validator.metagraph.axons))
        query_axons = [uid_to_axon[int(uid)] for uid in self.all_uids]
        synapse = ImageGenerating()
        synapse.request_dict = {"get_miner_info": True}
        bt.logging.info("Requesting miner info")
        responses = self.validator.dendrite.query(
            query_axons,
            synapse,
            deserialize=False,
            timeout=10,
        )
        responses = {
            uid: response.response_dict
            for uid, response in zip(self.all_uids, responses)
        }
        return responses

    def update_miners_identity(self):
        """
        1. Query model_name of available uids
        2. Update the available list
        """
        valid_miners_info = self.get_miner_info()
        if not valid_miners_info:
            bt.logging.warning("No active miner available. Skipping setting weights.")
        for uid, info in valid_miners_info.items():
            miner_state = self.all_uids_info.setdefault(
                uid,
                {
                    "scores": [],
                    "model_name": "",
                },
            )
            model_name = info.get("model_name", "")
            miner_state["total_volume"] = info.get("total_volume", 40)
            miner_state["min_stake"] = info.get("min_stake", 10000)
            miner_state["reward_scale"] = max(
                min(miner_state["total_volume"] ** 0.5 / 1000**0.5, 1), 0
            )
            volume_per_validator = get_volume_per_validator(
                self.validator.metagraph,
                miner_state["total_volume"],
                1.03,
                miner_state["min_stake"],
                False,
            )
            miner_state["rate_limit"] = volume_per_validator[self.validator.uid]

            if miner_state["model_name"] == model_name:
                continue
            miner_state["model_name"] = model_name
            miner_state["scores"] = []

        bt.logging.success("Updated miner identity")
        model_distribution = {}
        for uid, info in self.all_uids_info.items():
            model_distribution[info["model_name"]] = (
                model_distribution.get(info["model_name"], 0) + 1
            )
        # Remove all key type is str, keep only int from all_uids_info
        self.all_uids_info = {
            int(k): v for k, v in self.all_uids_info.items() if isinstance(k, int)
        }
        bt.logging.info(f"Model distribution: {model_distribution}")

    def get_miner_uids(self, model_name: str):
        available_uids = [
            int(uid)
            for uid in self.all_uids_info.keys()
            if self.all_uids_info[uid]["model_name"] == model_name
        ]
        return available_uids

    def update_scores(self, uids, rewards):
        for uid, reward in zip(uids, rewards):
            self.all_uids_info[uid]["scores"].append(reward)
            self.all_uids_info[uid]["scores"] = self.all_uids_info[uid]["scores"][-10:]

    def get_model_specific_weights(self, model_name, normalize=True):
        model_specific_weights = torch.zeros(len(self.all_uids))
        for uid in self.get_miner_uids(model_name):
            num_past_to_check = 10
            model_specific_weights[int(uid)] = (
                sum(self.all_uids_info[uid]["scores"][-num_past_to_check:])
                / num_past_to_check
            )
        model_specific_weights = torch.clamp(model_specific_weights, 0, 1)
        if normalize:
            tensor_sum = torch.sum(model_specific_weights)
            # Normalizing the tensor
            if tensor_sum > 0:
                model_specific_weights = model_specific_weights / tensor_sum
        return model_specific_weights
