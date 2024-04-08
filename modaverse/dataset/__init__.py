import torch
from torch.utils.data import DataLoader, DistributedSampler

from .caption_dataset import CaptionDataset
from .multi_modal_dataset import MultiModalDataset

__all__ = ['load_dataset', 'CaptionDataset', 'MultiModalDataset']


def load_dataset(dataset_cfgs: dict, ds_cfgs: dict, split: str = 'train'):
    dataset_cfgs = dataset_cfgs[split]
    dataset = MultiModalDataset(dataset_cfgs['instruction_path'],
                                media_root=dataset_cfgs['media_root'])
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_sz = world_size * ds_cfgs['train_micro_batch_size_per_gpu']
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_sz,
        sampler=sampler,
        num_workers=ds_cfgs['num_workers'],
        pin_memory=True,
    )

    return dataset, data_loader
