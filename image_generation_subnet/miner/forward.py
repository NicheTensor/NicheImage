import requests
from image_generation_subnet.protocol import ImageGenerating


def set_info(self):
    # Set information of miner
    # Currently only model name is set
    response = get_model_name(self)
    miner_info = {
        "model_name": response["model_name"],
        "total_volume": self.config.total_volume,
        "size_preference_factor": self.config.size_preference_factor,
        "min_stake": self.config.min_stake,
        "volume_per_validator": self.volume_per_validator,
    }
    return miner_info


async def generate(self, synapse: ImageGenerating) -> ImageGenerating:
    import httpx

    data = synapse.deserialize()

    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(
            self.config.generate_endpoint, headers=headers, json=data, timeout=60
        )
    synapse = synapse.copy(update=response.json())
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
