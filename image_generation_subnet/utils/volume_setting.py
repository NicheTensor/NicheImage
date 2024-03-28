import bittensor as bt
import torch


def get_volume_per_validator(
    metagraph,
    total_volume: int,
    size_preference_factor: float,
    min_stake: int,
    log: bool = True,
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
        if log:
            bt.logging.info(f"Volume for {uid}-validator: {metagraph.total_stake[uid]}")

    return volume_per_validator
