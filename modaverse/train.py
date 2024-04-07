import argparse
import json
import logging
import os
import os.path as osp
import shutil
import time

import deepspeed
import torch
from pjtools.configurator import AutoConfigurator
from tqdm import tqdm

from modaverse.dataset import load_dataset
from modaverse.model import DeepSpeedAgent, ModaVerse
from modaverse.utils import mkdir_if_not_exist


def parse_args():
    parser = argparse.ArgumentParser(description='Train Modaverse')
    parser.add_argument('--cfg-path',
                        type=str,
                        default='configs/base.py',
                        help='Path to the config file')
    parser.add_argument('--local_rank',
                        default=0,
                        type=int,
                        help='Local rank for distributed training')
    return parser.parse_args()


def init_dist(cfgs):
    cfgs['dist'] = dict(
        master_ip=os.getenv('MASTER_ADDR', 'localhost'),
        master_port=os.getenv('MASTER_PORT', '29500'),
        world_size=int(os.getenv('WORLD_SIZE', 1)),
        local_rank=int(os.getenv('LOCAL_RANK', 0)) % torch.cuda.device_count(),
    )
    device = cfgs['dist']['local_rank'] % torch.cuda.device_count()
    torch.cuda.set_device(device)
    deepspeed.init_distributed(
        dist_backend=cfgs.training_configs.deepspeed_cfg.backend)


def main(args):
    cfgs = AutoConfigurator.fromfile(args.cfg_path)
    training_cfgs = cfgs.training_configs
    with open(training_cfgs.deepspeed_cfg.path, 'r') as f:
        ds_cfgs = json.load(f)
    init_dist(cfgs)
    saving_path = osp.join(
        training_cfgs.saving_root,
        cfgs.model_configs.name + f'-{time.strftime("%Y%m%d%H%M%S")}')
    training_cfgs.saving_root = saving_path
    mkdir_if_not_exist(training_cfgs.saving_root)

    if cfgs.dist['local_rank'] == 0:
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
            level=logging.DEBUG,
            filename=osp.join(training_cfgs.saving_root, 'train.log'),
            filemode='w',
        )
        logging.info('Training Modaverse')

        shutil.copyfile(args.cfg_path,
                        osp.join(training_cfgs.saving_root, 'config.py'))
        logging.info(f'Using Configuration: {args.cfg_path}')

    dataset, data_loader = load_dataset(cfgs.dataset_configs, ds_cfgs, 'train')
    if cfgs.dist['local_rank'] == 0:
        dataset.diagnose()

    num_samples = len(dataset)
    length = training_cfgs.epochs * num_samples // cfgs.dist[
        'world_size'] // ds_cfgs['train_micro_batch_size_per_gpu']
    total_steps = training_cfgs.epochs * num_samples // ds_cfgs[
        'train_batch_size']

    model = ModaVerse(cfgs)
    state = torch.load('.checkpoints/init.pt', map_location='cuda')
    model.load_state_dict(state, strict=False)
    ds_cfgs['scheduler']['params']['total_num_steps'] = total_steps
    ds_cfgs['scheduler']['params']['warmup_num_steps'] = max(
        10, int(total_steps * training_cfgs.warmup_rate))
    training_agent = DeepSpeedAgent(model, cfgs, ds_cfgs)
    torch.distributed.barrier()
    ckpt = training_cfgs.checkpointer

    pbar = tqdm(total=length)
    current_step = 0
    for e in tqdm(range(training_cfgs.epochs)):
        for batch in data_loader:
            training_agent.train_model(batch,
                                       current_step=current_step,
                                       pbar=pbar)
            current_step += 1
            if ckpt.type == 'iteration' and current_step % ckpt.interval == 0:
                training_agent.save_model()
        if ckpt.type == 'epoch' and ckpt.interval % (e + 1) == 0:
            training_agent.save_model()

    torch.distributed.barrier()
    training_agent.save_model()


if __name__ == '__main__':
    args = parse_args()
    main(args)
