import json

import wandb
from torch.utils.tensorboard import SummaryWriter


class Writer:

    def __init__(self, config, save_path=None):
        self.backend = config.training_configs.report_backend.type
        if self.backend == 'tensorboard':
            self.writer = SummaryWriter(save_path)
        elif self.backend == 'wandb':
            wandb.init(project=config.model_configs.name,
                       config=self.to_serializable_dict(config.to_dict()))
            self.writer = wandb
        else:
            raise ValueError(f'Unsupported backend: {self.backend}')

    def log(self, data, step=None):
        if self.backend == 'tensorboard':
            for key, value in data.items():
                self.writer.add_scalar(key, value, step)
        elif self.backend == 'wandb':
            if step is not None:
                data.update({'step': step})
            wandb.log(data)

    def close(self):
        if self.backend == 'tensorboard':
            self.writer.close()

    def to_serializable_dict(self, config):
        serializable_dict = {}
        for key, value in config.items():
            try:
                json.dumps(value)
                serializable_dict[key] = value
            except (TypeError, OverflowError):
                pass
        return serializable_dict
