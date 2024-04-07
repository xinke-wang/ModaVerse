# ModaVerse: Efficiently Transforming Modalities with LLMs

<a href='https://xinke-wang.github.io/modaverse'><img src='https://img.shields.io/badge/Demo-Page-purple'></a>
<a href='https://arxiv.org/pdf/2401.06395'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a>
![License](https://img.shields.io/badge/License-BSD-blue.svg)

![ModaVerse](https://github.com/xinke-wang/ModaVerse/assets/45810070/0827efb2-5810-4133-9b4c-e70cabdd0f18)

🎆🎆🎆 **Visit our online demo [here](https://xinke-wang.github.io/modaverse).**

## TODO

- \[x\] 2024.04.07: Release the code of ModaVerse with version `ModaVerse-7b-v0`
- \[ \] Customize Diffusion Model Zoo
- \[ \] Add step-by-step data preparation instructions & instrution dataset
- \[ \] Training with custom data
- \[ \] Instruction generation scripts
- \[ \] Updating ModaVerse in versions with different setting

## Installation

```bash
conda create -n modaverse python=3.9 -y
conda activate modaverse
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

git clone --recursive https://github.com/xinke-wang/ModaVerse.git

cd ModaVerse
pip install -r requirements.txt
pip install -e .

rm -rf ImageBind/requirements.txt
cp requirements.txt ImageBind/requirements.txt

cd ImageBind
pip install -e .
cd ..
```

## Prepare Pretrained Models

```bash
mkdir .checkpoints && cd .checkpoints
```

Follow these [instructions](https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md) to obtain and apply Vicuna's `7b-v0` delta weights to the LLaMA pretrained model.

Then, download the ModaVerse pretrained model from one of the following sources:

| Model           | Foundation LLM | HuggingFace | GoogleDrive | Box |
|-----------------|----------------|-------------|-------------|-----|
| ModaVerse-7b-v0 |Vicuna-7b-V0    |[Model](https://huggingface.co/xinke-wang/ModaVerse-7b-v0)| [Model](https://drive.google.com/drive/folders/1W5FyAmgR-lIacHPjD4ma4yi5SIWSF6sO?usp=sharing) | [Model](https://universityofadelaide.app.box.com/s/x1mdcbpb026b0hjdp6zy56ljxloaugy6) |
| ModaVerse-chat | Coming Soon | | |

Next, manually download the [ImageBind](https://github.com/facebookresearch/ImageBind) model, or it will be automatically downloaded to `.checkpoints/` when running the ModaVerse code. Finally, place all the weights in the `.checkpoints/` folder, following the structure below:

```text
.checkpoints/
    ├── 7b_v0
    │   ├── config.json
    │   ├── generation_config.json
    │   ├── model-00001-of-00003.safetensors
    │   ├── model-00002-of-00003.safetensors
    │   ├── model-00003-of-00003.safetensors
    │   ├── model.safetensors.index.json
    │   ├── special_tokens_map.json
    │   ├── tokenizer_config.json
    │   └── tokenizer.model
    ├── imagebind_huge.pth
    └── ModaVerse-7b
        ├── added_tokens.json
        ├── config.json
        ├── config.py
        ├── pytorch_model.pt
        ├── special_tokens_map.json
        ├── tokenizer_config.json
        └── tokenizer.model
```

## Usage

A simple example of using the model is as follows:

```python
from modaverse.api import ModaVerseAPI

ModaVerse = ModaVerseAPI()

# Only Text Instruction
text_instruction = 'Please generate an audio that a dog is barking.'
ModaVerse(text_instruction)

# With Multi-modal Input
text_instruction = 'Please generate an audio of the sound for the animal in the image.'
ModaVerse(text_instruction, ['assets/media/image/1.jpg'])
```

The output is saved in the `output` folder by default.

Running inference with fully equipped generators for three-modality diffusion models may require at least 40 GB of GPU memory. If you lack sufficient memory, consider setting `meta_response_only=True` to receive only the meta response from the model. And customize the parser and generator to fit your needs.

```python
ModaVerse = ModaVerseAPI(meta_response_only=True)
```

## Running the Demo

```bash
python demo.py
```

![image](https://github.com/xinke-wang/ModaVerse/assets/45810070/8873348a-b322-43f2-b207-bfe435b3fc38)

## Citation

If you find ModaVerse useful in your research or applications, please consider cite:

```bibtex
@article{wang2024modaverse,
  title={ModaVerse: Efficiently Transforming Modalities with LLMs},
  author={Wang, Xinyu and Zhuang, Bohan and Wu, Qi},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```

## Acknowledgements

We would like to thank the authors of the following repositories for their valuable contributions:
[ImageBind](https://github.com/facebookresearch/ImageBind),
[MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4),
[Vicuna](https://github.com/lm-sys/FastChat),
[Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img),
[AudioLDM](https://github.com/haoheliu/AudioLDM),
[NextGPT](https://github.com/NExT-GPT/NExT-GPT),
[VideoFusion](https://huggingface.co/ali-vilab/text-to-video-ms-1.7b)
