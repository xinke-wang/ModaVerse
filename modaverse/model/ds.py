import datetime
import logging
import random
import types
from collections import OrderedDict

import deepspeed
import numpy as np
import torch

from modaverse.utils.writer import Writer


class DeepSpeedAgent:

    def __init__(self, model, cfgs, dscfgs, random_seed=42):
        super(DeepSpeedAgent, self).__init__()
        self.cfgs = cfgs
        self.dscfgs = dscfgs
        self.model = model
        self.train_cfg = cfgs.training_configs
        self.log_interval = self.train_cfg.report_backend.log_interval
        self.save_path = self.train_cfg.saving_root
        logging.info('Trainable Parameters:')
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                logging.info(f'{n}: {p.shape}')

        if self.cfgs.dist['local_rank'] == 0:
            self.writer = Writer(self.cfgs, self.save_path)

        # Random seed
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)

        self.ds_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            config_params=self.dscfgs,
            dist_init_required=True,
            args=types.SimpleNamespace(**cfgs.to_dict()))

    def train_model(self, batch, current_step=0, pbar=None):
        self.ds_engine.module.train()

        loss, acc = self.ds_engine(batch)
        self.ds_engine.backward(loss)
        self.ds_engine.step()

        if self.cfgs.dist[
                'local_rank'] == 0 and current_step % self.train_cfg.report_backend.iterval == 0:  # noqa
            self.writer.log(
                {
                    'loss': loss,
                    'acc': acc,
                    'lr': self.optimizer.get_lr()
                }, current_step)

        pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; '
                             f'token_acc: {round(acc * 100, 2)}')
        pbar.update(1)

        if self.cfgs.dist['local_rank'] == 0 and self.cfgs[
                'log_path'] and current_step % self.cfgs['logging_step'] == 0:
            rate = pbar.format_dict['rate']
            remaining = (pbar.total -
                         pbar.n) / rate if rate and pbar.total else 0

            remaining = str(datetime.timedelta(seconds=remaining))
            logging.info(
                '[!] progress: %s; remaining time: %s; loss: %s; '
                'token_acc: %s', round(pbar.n / pbar.total, 5), remaining,
                round(loss.item(), 4), round(acc * 100, 2))

        acc *= 100
        return acc

    def save_model(self):
        checkpoint = OrderedDict()
        for k, v in self.ds_engine.module.named_parameters():
            if v.requires_grad:
                checkpoint[k] = v
        torch.save(checkpoint, f'{self.save_path}/pytorch_model.pt')
        self.model.tokenizer.save_pretrained(self.save_path)
        self.model.LLM.config.save_pretrained(self.save_path)
        logging.info(f'[!] save model into {self.save_path}')
