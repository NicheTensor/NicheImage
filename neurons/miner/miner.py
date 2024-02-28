import time
import typing
import bittensor as bt
from image_generation_subnet.base.miner import BaseMinerNeuron
import image_generation_subnet


class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        self.validator_logs = {}
        self.miner_info = image_generation_subnet.miner.set_info(self)
        self.axon.attach(
            forward_fn=self.temp_forward,
            blacklist_fn=self.temp_blacklist,
            priority_fn=self.temp_priority,
        )

    async def forward(
        self, synapse: image_generation_subnet.protocol.ImageGenerating
    ) -> image_generation_subnet.protocol.ImageGenerating:
        
        bt.logging.info(f"{synapse.name}: {synapse}")

        if synapse.prompt:
            image = image_generation_subnet.miner.generate(
                self, synapse.prompt, synapse.seed, synapse.pipeline_params
            )
            synapse.image = image

        if synapse.request_dict:
            synapse.response_dict = self.miner_info
            bt.logging.info(f"Response dict: {self.miner_info}")
        return synapse

    async def blacklist(
        self, synapse: image_generation_subnet.protocol.ImageGenerating
    ) -> typing.Tuple[bool, str]:
        
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
            self, uid=validator_uid, stake=stake
        ):
            bt.logging.trace(
                f"Blacklisting {validator_uid}-validator for exceeding the limit"
            )
            return True, "Limit exceeded"

        return False, "All passed!"

    async def priority(
        self, synapse: image_generation_subnet.protocol.ImageGenerating
    ) -> float:
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

    async def temp_forward(
        self, synapse: image_generation_subnet.protocol.NicheImageProtocol
    ) -> image_generation_subnet.protocol.NicheImageProtocol:
        return await self.forward(synapse)
    
    async def temp_blacklist(
        self, synapse: image_generation_subnet.protocol.NicheImageProtocol
    ) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)
    
    async def temp_priority(
        self, synapse: image_generation_subnet.protocol.NicheImageProtocol
    ) -> float:
        return await self.priority(synapse)


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info("Miner running...", time.time())
            time.sleep(5)
