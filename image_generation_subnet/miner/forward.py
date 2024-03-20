import requests
import httpx
import bittensor as bt


def set_info(self):
    # Set information of miner
    # Currently only model name is set
    miner_info = {}
    response = get_model_name(self)
    miner_info["model_name"] = response["model_name"]
    return miner_info


async def generate(self, synapse: bt.Synapse) -> bt.Synapse:
    data = synapse.deserialize()
    async with httpx.AsyncClient() as client:
        response = await client.post(
            self.config.generate_endpoint, json=data, timeout=synapse.timeout
        )
    synapse = synapse.miner_update(update=response.json())
    return synapse


def get_model_name(self):
    # Get running model's name
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }
    response = requests.get(self.config.info_endpoint, headers=headers)
    response = response.json()
    return response
