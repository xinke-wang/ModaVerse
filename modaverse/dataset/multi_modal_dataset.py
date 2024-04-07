import copy
import json
import os.path as osp

from torch.utils.data import Dataset
from tqdm import tqdm


class MultiModalDataset(Dataset):

    def __init__(self, instruction_path: str, media_root: str):
        with open(instruction_path, 'r') as f:
            self.instructions = json.load(f)
        self.media_root = media_root

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx):
        instruction = copy.deepcopy(self.instructions[idx])
        media = instruction['media']
        instruction['media'] = {
            key: osp.join(self.media_root, value) if value != 'none' else value
            for key, value in media.items()
        }

        return instruction

    def diagnose(self):
        print('Dataset Diagnose...')
        modality_instructions = {}
        original_count = len(self.instructions)
        instructions_to_keep = []

        for ins in tqdm(self.instructions):
            input_modality = []
            file_exists = True

            for k, v in ins['media'].items():
                if v != 'none':
                    input_modality.append(k)
                    if not osp.exists(osp.join(self.media_root, v)):
                        file_exists = False
                        break

            if len(ins['instruction']) == 0:
                file_exists = False
            elif file_exists:
                input_modality.append('text')
                input_modality = '+'.join(input_modality)
                output_modality = ins['meta_response']['modality']
                pair = f'{input_modality}->{output_modality}'

                if pair not in modality_instructions:
                    modality_instructions[pair] = 1
                else:
                    modality_instructions[pair] += 1

                instructions_to_keep.append(ins)

        self.instructions = instructions_to_keep
        removed_count = original_count - len(self.instructions)

        print(f'Removed {removed_count} missing files or invalid data.')
        print(f'Total number of instructions: {len(self.instructions)}')
        for key, value in modality_instructions.items():
            print(f'{key}: {value}')
