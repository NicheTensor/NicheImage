import timm
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import torch
from typing import List
from generation_models.utils import base64_to_pil_image
import imagehash
from io import BytesIO
import copy

def pil_convert(image: Image.Image, format: str) -> Image.Image:
    image_stream = BytesIO()
    image.save(image_stream, format=format)
    image = Image.open(image_stream)
    return image

class CosineSimilarityReward(nn.Module):
    def __init__(
        self,
        model_name="vit_base_patch16_clip_384.laion2b_ft_in12k_in1k",
        threshold=0.6,
    ):
        super(CosineSimilarityReward, self).__init__()
        self.threshold = threshold
        self.model, self.transforms = self.get_model(model_name)

    def get_model(self, model_name):
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        model.eval()
        model = model.cuda()
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        return model, transforms

    @torch.inference_mode()
    def forward(
        self, validator_image: Image.Image, miner_image: Image.Image, binary=True
    ) -> float:
        validator_vec = self.model(self.transforms(validator_image).unsqueeze(0).cuda())
        image_vec = self.model(self.transforms(miner_image).unsqueeze(0).cuda())
        cosine_similarity = F.cosine_similarity(validator_vec, image_vec)
        if binary:
            return float(cosine_similarity.item() > self.threshold)
        return float(cosine_similarity.item())

    def get_reward(
        self, validator_image: Image.Image, batched_miner_images: List[str]
    ) -> List[float]:
        rewards = []
        if not isinstance(validator_image, Image.Image):
            validator_image = base64_to_pil_image(validator_image)
        for miner_image in batched_miner_images:
            if not miner_image:
                reward = False
            else:
                try:
                    if not isinstance(miner_image, Image.Image):
                        miner_image = base64_to_pil_image(miner_image)
                    miner_image_format = miner_image.format
                    temp_validator_image = copy.deepcopy(validator_image)
                    # convert validator image to the same format as miner image
                    temp_validator_image = pil_convert(temp_validator_image, miner_image_format)
                    
                except Exception:
                    print("Corrupted miner image", flush=True)
                    reward = False
                    rewards.append(reward)
                    continue
                nsfw_check = self.nsfw_filter(validator_image, miner_image)
                if nsfw_check:
                    reward = -5 if nsfw_check == 2 else 0
                else:
                    reward = self.matching_image(miner_image, validator_image)
            rewards.append(reward)
        return rewards

    def get_black_hash(self, H, W) -> str:
        image = Image.new("RGB", (W, H), color="black")
        return str(imagehash.average_hash(image, hash_size=8))

    def matching_image(
        self, miner_image: Image.Image, validator_image: Image.Image
    ) -> bool:
        cosine_similarity_score = self.forward(validator_image, miner_image)
        print("Cosine Similarity Score:", cosine_similarity_score, flush=True)
        return cosine_similarity_score

    def nsfw_filter(
        self, validator_image: Image.Image, miner_image: Image.Image
    ) -> bool:
        W, H = validator_image.size
        validator_hash = imagehash.average_hash(validator_image, hash_size=6)
        miner_hash = imagehash.average_hash(miner_image, hash_size=6)
        validator_hash = str(validator_hash)
        miner_hash = str(miner_hash)
        black_hash = self.get_black_hash(H, W)
        if validator_hash != black_hash and miner_hash == black_hash:
            return 1
        if validator_hash == black_hash and miner_hash != black_hash:
            return 2
        return 0
