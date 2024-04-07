import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from peft import get_peft_model
from transformers import LlamaForCausalLM, LlamaTokenizer

from modaverse.utils import format_paths, get_media_type, split_media


class ModaVerse(nn.Module):
    """ModaVerse model implementation"""

    Modality_Map = {
        'video': {
            'modality': ModalityType.VISION,
            'loader': data.load_and_transform_video_data
        },
        'audio': {
            'modality': ModalityType.AUDIO,
            'loader': data.load_and_transform_audio_data
        },
        'image': {
            'modality': ModalityType.VISION,
            'loader': data.load_and_transform_vision_data
        }
    }

    def __init__(self, configs):
        super(ModaVerse, self).__init__()
        self.device = torch.cuda.current_device()
        self.cfgs = configs
        self.model_cfgs = configs.model_configs
        self.train_cfgs = configs.training_configs
        self.modaverse_cfgs = configs.model_configs.modaverse
        # Initialize imagebind encoder
        self.imagebind_encoder = imagebind_model.imagebind_huge(
            pretrained=True)
        for _, param in self.imagebind_encoder.named_parameters():
            param.requires_grad = False
        self.imagebind_encoder.eval()
        # Initialize foundation LLM
        llm_path = self.model_cfgs.foundation_llm.checkpoint
        self.LLM = LlamaForCausalLM.from_pretrained(llm_path)
        self.LLM = get_peft_model(self.LLM, self.train_cfgs.lora_config)
        # Initialize llama tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(llm_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        self.tokenizer.add_tokens([self.modaverse_cfgs.modality_begin_token])
        self.tokenizer.add_tokens([self.modaverse_cfgs.modality_end_token])
        for flag in self.modaverse_cfgs.modality_flags:
            self.tokenizer.add_tokens([flag])
        self.target_padding = self.modaverse_cfgs.target_padding
        self.LLM.resize_token_embeddings(len(self.tokenizer))
        self.input_embeddings = self.LLM.get_input_embeddings()
        self.max_length = self.modaverse_cfgs.max_length
        self.stopping_flag = self.tokenizer.decode(
            self.modaverse_cfgs.stopping_token)
        if len(self.train_cfgs.force_training_layers) > 0:
            print('Forcing training on layers: ')
            for name, param in self.LLM.named_parameters():
                if any(layer in name
                       for layer in self.train_cfgs.force_training_layers):
                    print(name)
                    param.requires_grad = True
        self.LLM.print_trainable_parameters()
        # Initialize linear projection layer
        self.linear_proj = nn.Linear(self.model_cfgs.imagebind.hidden_size,
                                     self.LLM.config.hidden_size)
        # Prompt template
        self.prompt_config = configs.prompt_configs
        with open(self.prompt_config.path, 'r') as f:
            self.prompt_template = f.read()

    def encode_media(self, media_path: str, media_type: Optional[str] = None):
        if media_type is None:
            media_type = get_media_type(media_path)
        if media_type not in self.Modality_Map:
            raise ValueError(f'Unsupported media_type: {media_type}')

        modality = self.Modality_Map[media_type]['modality']
        loader = self.Modality_Map[media_type]['loader']

        inputs = {modality: loader([media_path], self.device)}
        inputs = {key: inputs[key].to(self.LLM.dtype) for key in inputs}

        with torch.no_grad():
            embeddings = self.imagebind_encoder(inputs)
            media_embeds = embeddings[modality]

        media_embeds = self.linear_proj(media_embeds).unsqueeze(1)

        media_bos = self.tokenizer(
            self.model_cfgs.modaverse.modality_begin_token,
            add_special_tokens=False,
            return_tensors='pt').to(self.device)
        media_eos = self.tokenizer(
            self.model_cfgs.modaverse.modality_end_token,
            add_special_tokens=False,
            return_tensors='pt').to(self.device)

        bos_embeds = self.LLM.model.model.embed_tokens(
            media_bos.input_ids).expand(1, -1, -1)
        eos_embeds = self.LLM.model.model.embed_tokens(
            media_eos.input_ids).expand(1, -1, -1)

        return torch.cat([bos_embeds, media_embeds, eos_embeds], dim=1)

    def encode_text(self, text: str):
        tokens = self.tokenizer(text.strip(),
                                add_special_tokens=False,
                                return_tensors='pt').to(self.device)
        text_embeds = self.LLM.model.model.embed_tokens(
            tokens.input_ids).expand(1, -1, -1)

        return text_embeds

    def embed_prompt_template(self,
                              text_instructions: str,
                              media: List,
                              caption: Optional[str] = None):
        media = format_paths(media)
        prompt = copy.deepcopy(self.prompt_template)
        prompt = prompt.replace(self.prompt_config.media_placeholder, media)
        prompt = prompt.replace(self.prompt_config.instruction_placeholder,
                                text_instructions)
        if caption is None:
            return prompt
        else:
            caption = caption + ' ' + self.stopping_flag
            prompt = prompt + caption
            return prompt, caption

    def tokenize(self, prompt: str, target: Optional[str] = None):
        splits = split_media(prompt)
        inputs = []

        # <bos>
        bos = self.tokenizer(self.tokenizer.bos_token,
                             add_special_tokens=False,
                             return_tensors='pt').to(self.device)
        bos = self.LLM.model.model.embed_tokens(bos.input_ids).expand(
            1, -1, -1)
        inputs.append(bos)

        # multi-modal content
        for split in splits:
            modality, content = split
            if modality != 'text':
                inputs.append(self.encode_media(content, modality))
            else:
                text_embeds = self.encode_text(content)
                inputs.append(text_embeds)

        inputs = torch.cat(inputs, dim=1)
        attn_mask = torch.ones(inputs.shape[:2],
                               dtype=torch.long).to(self.device)

        if target is None:
            return inputs.to(self.LLM.dtype), attn_mask
        else:
            target = self.tokenizer(target,
                                    add_special_tokens=False,
                                    return_tensors='pt').to(
                                        self.device).input_ids
            pad = torch.ones(
                (target.shape[0], inputs.shape[1] - target.shape[1]),
                dtype=torch.long).to(target.device).fill_(self.target_padding)
            target = torch.cat([pad, target], dim=1)

            return inputs.to(self.LLM.dtype), attn_mask, target

    def generate(self, text_instructions: str, media: List = []):
        prompt = self.embed_prompt_template(text_instructions, media)
        inputs_embeds, _ = self.tokenize(prompt)

        outputs = self.LLM.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=self.modaverse_cfgs.max_new_tokens,
            do_sample=self.modaverse_cfgs.do_sample,
            top_p=self.modaverse_cfgs.top_p,
            temperature=self.modaverse_cfgs.temperature,
            use_cache=self.modaverse_cfgs.use_cache,
            stopping_criteria=self.modaverse_cfgs.stopping_criteria,
        )

        caption = self.tokenizer.batch_decode(outputs,
                                              skip_special_tokens=True)[0]

        return caption

    def forward(self, inputs: Dict):

        inputs_embeds, attention_mask, targets = self.get_training_inputs(
            inputs)

        outputs = self.LLM(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            labels=targets,
        )

        loss = outputs.loss

        predicts = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]
        labels = targets[:, 2:]

        # modality_gt = labels[torch.arange(labels.size(0)), (
        #     labels != self.target_padding).to(torch.long).argmax(dim=1)]
        # modality_gt = [
        #     self.model_cfgs.modaverse.modality_flags.index(
        #         self.tokenizer.decode(flag)) for flag in modality_gt
        # ]

        acc = (predicts.reshape(-1) == labels.reshape(-1)).to(torch.long)
        valid = (labels != self.target_padding).reshape(-1)
        acc = acc & valid
        acc = acc.sum().item() / (valid.sum().item() + 1)
        return loss, acc

    def get_training_inputs(self, inputs: Dict):
        instruction = inputs['instruction']
        media = [[item for item in group if item != 'none']
                 for group in zip(*inputs['media'].values())]
        meta_response = [
            f'Modality: [{modality.upper()}] Prompt: {prompt}'
            for modality, prompt in zip(inputs['meta_response']['modality'],
                                        inputs['meta_response']['prompt'])
        ]

        input_embeds, input_attns, targets = [], [], []
        for i, m, r in zip(instruction, media, meta_response):
            prompt, caption = self.embed_prompt_template(i, m, r)
            embed, attn, target = self.tokenize(prompt, caption)
            input_embeds.append(embed)
            input_attns.append(attn)
            targets.append(target)
        input_embeds, input_attns, targets = self.pad_input_embeds(
            input_embeds, input_attns, targets)

        input_embeds = input_embeds[:, :self.max_length, :]
        input_attns = input_attns[:, :self.max_length]
        targets = targets[:, :self.max_length]

        assert input_embeds.shape[0] == input_attns.shape[0] == targets.shape[
            0]  # noqa
        assert input_embeds.shape[0] == input_attns.shape[0] == targets.shape[
            0]  # noqa

        return input_embeds, input_attns, targets

    def pad_input_embeds(self, embeds: List, attns: List, targets: List):
        max_len = max(embed.shape[1] for embed in embeds)
        pad_token_embed = self.tokenizer(self.tokenizer.pad_token,
                                         add_special_tokens=False,
                                         return_tensors='pt').to(self.device)
        pad_token_embed = self.LLM.model.model.embed_tokens(
            pad_token_embed.input_ids).squeeze(0)

        padded_embeds, padded_attns, padded_targets = [], [], []
        for embed, attn, target in zip(embeds, attns, targets):
            pad_length = max_len - embed.shape[1]
            if pad_length > 0:
                embed_pad = pad_token_embed.unsqueeze(0).expand(
                    embed.shape[0], pad_length, -1)
                embed = torch.cat([embed, embed_pad], dim=1)
                attn_pad = torch.zeros(attn.shape[0],
                                       pad_length,
                                       dtype=attn.dtype).to(attn.device)
                attn = torch.cat([attn, attn_pad], dim=1)
                target_pad = torch.ones(target.shape[0],
                                        pad_length,
                                        dtype=target.dtype).to(
                                            target.device).fill_(
                                                self.target_padding)
                target = torch.cat([target, target_pad], dim=1)

            padded_embeds.append(embed)
            padded_attns.append(attn)
            padded_targets.append(target)

        return torch.cat(padded_embeds,
                         dim=0), torch.cat(padded_attns,
                                           dim=0), torch.cat(padded_targets,
                                                             dim=0)
