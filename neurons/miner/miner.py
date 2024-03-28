import time
from typing import Tuple, TypeVar
import bittensor as bt
from image_generation_subnet.base.miner import BaseMinerNeuron
import image_generation_subnet
from image_generation_subnet.protocol import ImageGenerating, TextGenerating
import traceback

T = TypeVar("T", bound=bt.Synapse)


class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        self.validator_logs = {}
        self.volume_per_validator = (
            image_generation_subnet.utils.volume_setting.get_volume_per_validator(
                self.metagraph,
                self.config.miner.total_volume,
                self.config.miner.size_preference_factor,
                self.config.miner.min_stake,
            )
        )
        self.miner_info = image_generation_subnet.miner.set_info(self)
        self.num_processing_requests = 0
        bt.logging.info(f"Miner info: {self.miner_info}")

    async def forward_image(self, synapse: ImageGenerating) -> ImageGenerating:
        if synapse.request_dict:
            return await self.forward_info(synapse)
        self.num_processing_requests += 1
        bt.logging.info(
            f"Processing {self.num_processing_requests} requests, synapse prompt: {synapse.prompt}"
        )
        synapse = await image_generation_subnet.miner.generate(self, synapse)
        self.num_processing_requests -= 1
        return synapse

    async def forward_info(self, synapse: ImageGenerating) -> ImageGenerating:
        synapse.response_dict = self.miner_info
        bt.logging.info(f"Response dict: {self.miner_info}")
        return synapse

    async def forward_text(self, synapse: TextGenerating) -> TextGenerating:
        if synapse.request_dict:
            return await self.forward_info(synapse)
        self.num_processing_requests += 1
        bt.logging.info(
            f"Processing {self.num_processing_requests} requests, synapse input: {synapse.prompt_input}"
        )
        synapse = await image_generation_subnet.miner.generate(self, synapse)
        self.num_processing_requests -= 1
        return synapse

    async def blacklist(self, synapse: ImageGenerating) -> Tuple[bool, str]:
        bt.logging.info(f"synapse in blacklist {synapse}")
        try:
            if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
                # Ignore requests from unrecognized entities.
                bt.logging.trace(
                    f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Unrecognized hotkey"
            if (
                self.num_processing_requests
                >= self.config.miner.max_concurrent_requests
            ):
                bt.logging.trace(
                    f"Blacklisting {synapse.dendrite.hotkey} for exceeding the limit of concurrent requests"
                )
                return True, "Max concurrent requests exceeded"

            validator_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
            stake = self.metagraph.stake[validator_uid].item()

            if image_generation_subnet.miner.check_min_stake(
                stake, validator_uid, self.config.miner.min_stake
            ):
                bt.logging.trace(
                    f"Blacklisting {validator_uid}-validator has {stake} stake"
                )
                return True, "Not enough stake"
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
