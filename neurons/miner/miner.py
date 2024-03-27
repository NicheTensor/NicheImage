import time
from typing import Tuple, TypeVar
import bittensor as bt
from image_generation_subnet.base.miner import BaseMinerNeuron
import image_generation_subnet
from image_generation_subnet.protocol import ImageGenerating, TextGenerating
import torch
import traceback
T = TypeVar("T", bound=bt.Synapse)


class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        self.validator_logs = {}
        self.volume_per_validator = self.get_volume_per_validator(
            self.metagraph,
            self.config.miner.total_volume,
            self.config.miner.size_preference_factor,
            self.config.miner.min_stake,
        )
        self.miner_info = image_generation_subnet.miner.set_info(self)
        bt.logging.info(f"Miner info: {self.miner_info}")

    def get_volume_per_validator(
        self,
        metagraph,
        total_volume: int,
        size_preference_factor: float,
        min_stake: int,
    ) -> dict:
        valid_stakes = [
            stake for stake in metagraph.total_stake.tolist() if stake >= min_stake
        ]
        valid_uids = [
            uid
            for uid, stake in enumerate(metagraph.total_stake.tolist())
            if stake >= min_stake
        ]
        if not valid_stakes:
            bt.logging.warning(
                (
                    f"No validators with stake greater than {min_stake} found. "
                    "Assigning equal volume to all validators."
                    f"Total volume: {total_volume}"
                    f"Metagraph stake: {metagraph.total_stake.tolist()}"
                )
            )
        valid_stakes = torch.tensor(valid_stakes) + 1e-4
        prefered_valid_stakes = valid_stakes * size_preference_factor
        normalized_prefered_valid_stakes = (
            prefered_valid_stakes / prefered_valid_stakes.sum()
        )
        volume_per_validator = total_volume * normalized_prefered_valid_stakes
        volume_per_validator = torch.ceil(volume_per_validator)
        volume_per_validator = dict(zip(valid_uids, volume_per_validator.tolist()))
        for uid, volume in volume_per_validator.items():
            if metagraph.total_stake[uid] >= 10000:
                volume_per_validator[uid] = max(3, volume)
            bt.logging.info(f"Volume for {uid}-validator: {volume}")

        return volume_per_validator

    async def forward_image(self, synapse: ImageGenerating) -> ImageGenerating:
        if synapse.request_dict:
            return await self.forward_info(synapse)
        bt.logging.info(
            f"synapse prompt: {synapse.prompt}, pipeline_type: {synapse.pipeline_type}"
        )
        synapse = await image_generation_subnet.miner.generate(self, synapse)
        return synapse
    async def forward_info(self, synapse: ImageGenerating) -> ImageGenerating:
        synapse.response_dict = self.miner_info
        bt.logging.info(f"Response dict: {self.miner_info}")
        return synapse

    async def forward_text(self, synapse: TextGenerating) -> TextGenerating:
        if synapse.request_dict:
            return await self.forward_info(synapse)
        bt.logging.info(f"synapse prompt: {synapse.prompt_input}")
        synapse = await image_generation_subnet.miner.generate(self, synapse)
        return synapse
    
    async def blacklist(self, synapse: ImageGenerating) -> Tuple[bool, str]:
        bt.logging.info(f"synapse in blacklist {synapse}")
        try:
            if "get_miner_info" in synapse.request_dict:
                return False, "Getting info request, passed!"
            if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
                # Ignore requests from unrecognized entities.
                bt.logging.trace(
                    f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Unrecognized hotkey"

            validator_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
            stake = self.metagraph.stake[validator_uid].item()

            if image_generation_subnet.miner.check_min_stake(
                stake, validator_uid, self.config.miner.min_stake
            ):
                bt.logging.trace(
                    f"Blacklisting {validator_uid}-validator has {stake} stake"
                )
                return True, "Not enough stake"

            if "get_miner_info" in synapse.request_dict:
                return False, "Getting info request, passed!"
            if image_generation_subnet.miner.check_limit(
                self,
                uid=validator_uid,
                stake=stake,
                volume_per_validator=self.volume_per_validator,
                interval=self.config.miner.limit_interval,
            ):
                bt.logging.trace(
                    f"Blacklisting {validator_uid}-validator for exceeding the limit"
                )
                return True, "Limit exceeded"

            return False, "All passed!"
        except Exception as e:
            bt.logging.error(f"Error in blacklist: {e}")
            traceback.print_exc()
            return False, "All passed!"
    async def blacklist_image(self, synapse: ImageGenerating) -> Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def blacklist_text(self, synapse: TextGenerating) -> Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def priority(self, synapse: ImageGenerating) -> float:
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        prirority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", prirority
        )
        return prirority


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info("Miner running...", time.time())
            try:
                miner.volume_per_validator = miner.get_volume_per_validator(
                    miner.metagraph,
                    miner.config.miner.total_volume,
                    miner.config.miner.size_preference_factor,
                    miner.config.miner.min_stake,
                )
            except Exception:
                pass
            time.sleep(60)
