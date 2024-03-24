import time
from typing import Tuple, TypeVar
import bittensor as bt
from image_generation_subnet.base.miner import BaseMinerNeuron
import image_generation_subnet
from image_generation_subnet.protocol import (
    ImageGenerating,
)
import torch

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
            stake for stake in metagraph.total_stake.tolist() if stake > min_stake
        ]
        valid_uids = [
            uid
            for uid, stake in enumerate(metagraph.total_stake.tolist())
            if stake > min_stake
        ]
        valid_stakes = torch.tensor(valid_stakes)
        prefered_valid_stakes = valid_stakes * size_preference_factor
        normalized_prefered_valid_stakes = (
            prefered_valid_stakes / prefered_valid_stakes.sum()
        )
        volume_per_validator = total_volume * normalized_prefered_valid_stakes
        volume_per_validator = torch.ceil(volume_per_validator)
        volume_per_validator = dict(zip(valid_uids, volume_per_validator.tolist()))

        return volume_per_validator

    async def forward(self, synapse: ImageGenerating) -> ImageGenerating:
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

    async def blacklist(self, synapse: ImageGenerating) -> Tuple[bool, str]:
        bt.logging.info(f"synapse in blacklist {synapse}")

        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        validator_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        stake = self.metagraph.stake[validator_uid].item()

        if image_generation_subnet.miner.check_min_stake(stake, validator_uid):
            bt.logging.trace(
                f"Blacklisting {validator_uid}-validator has {stake} stake"
            )
            return True, "Not enough stake"
        if image_generation_subnet.miner.check_limit(
            self,
            uid=validator_uid,
            stake=stake,
            volume_per_validator=self.volume_per_validator,
        ):
            bt.logging.trace(
                f"Blacklisting {validator_uid}-validator for exceeding the limit"
            )
            return True, "Limit exceeded"

        return False, "All passed!"

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
            time.sleep(5)
