import json

import gradio as gr

from modaverse.api import ModaVerseAPI

pretrained_model = '.checkpoints/ModaVerse-7b'
ModaVerse = ModaVerseAPI(model_path=pretrained_model)


def create_tab(title, examples_list):
    with gr.Tab(title):
        gr.Examples(examples=examples_list,
                    fn=process_input,
                    inputs=[text_input, image_input, audio_input, video_input],
                    outputs=[
                        meta_response, text_output, image_output, audio_output,
                        video_output
                    ])


def process_input(instruction, image, audio, video):
    if not instruction:
        meta_response = {
            'type': 'text',
            'content': 'Text Instruction Cannot Be Empty!'
        }
        final_responses = ['Text Instruction Cannot Be Empty!']
    else:
        media = []
        if image:
            media.append(image)
        if audio:
            media.append(audio)
        if video:
            media.append(video)
        meta_response, final_responses = ModaVerse(instruction, media)

    text_content = None
    image_path = None
    audio_path = None
    video_path = None

    for final_response in final_responses:
        if final_response['type'] == 'text':
            text_content = final_response['content']
        elif final_response['type'] == 'image':
            image_path = final_response['content']
        elif final_response['type'] == 'audio':
            audio_path = final_response['content']
        elif final_response['type'] == 'video':
            video_path = final_response['content']

    return meta_response, text_content, image_path, audio_path, video_path


with gr.Blocks() as demo:
    print('Launching ModaVerse Demo...')

    gr.HTML("""
        <h1 align="center" style="display: flex; flex-direction: row; justify-content: center; font-size: 25pt;">
            <img src='https://i.ibb.co/4ZVthHT/Moda-Verse.png' width="45" height="45" style="margin-right: 10px;">
            ModaVerse: Efficiently Transforming Modalities with LLMs
        </h1>
        <div align="center" style="display: flex;"><a href='https://github.com/xinke-wang/ModaVerse'><img src='https://img.shields.io/badge/Github-Code-blue'></a> &nbsp &nbsp  &nbsp  <a href='https://arxiv.org/pdf/2401.06395.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a></div>
    """)  # noqa

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown('## Inputs')
            with gr.Row():
                text_input = gr.Textbox(label='Text Input')
            with gr.Row():
                image_input = gr.Image(label='Image Upload', type='filepath')
                audio_input = gr.Audio(label='Audio Upload', type='filepath')
                video_input = gr.Video(label='Video Upload', type='filepath')

        with gr.Column(scale=1):
            gr.Markdown('## Meta Response')
            meta_response = gr.Textbox(label='Meta Output', interactive=False)

        with gr.Column(scale=1):
            gr.Markdown('## Results')
            with gr.Row():
                text_output = gr.Textbox(label='Text Output',
                                         interactive=False)
            with gr.Row():
                image_output = gr.Image(label='Image Output',
                                        interactive=False)
                audio_output = gr.Audio(label='Audio Output',
                                        interactive=False)
                video_output = gr.Video(label='Video Output',
                                        interactive=False)

    process_button = gr.Button('Process').click(
        fn=process_input,
        inputs=[text_input, image_input, audio_input, video_input],
        outputs=[
            meta_response, text_output, image_output, audio_output,
            video_output
        ])

    with open('assets/examples.json', 'r', encoding='utf-8') as f:
        examples_dict = json.load(f)

    for task, examples in examples_dict.items():
        create_tab(task, examples)

demo.launch(share=True)
