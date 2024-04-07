import json
import os
import re
import tempfile

import cv2
import numpy as np
import PIL

MEDIA_EXTENSIONS = {
    'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'],
    'video': ['.mp4', '.mov', '.avi', '.flv', '.mkv'],
    'audio': ['.mp3', '.wav', '.aac', '.flac']
}


def get_media_type(file_name):
    _, ext = os.path.splitext(file_name)
    ext = ext.lower()

    for media_type, extensions in MEDIA_EXTENSIONS.items():
        if ext in extensions:
            return media_type

    raise ValueError(f'Unsupported media type: {ext}')


def format_paths(media_paths):
    media_counts = {'image': 0, 'video': 0, 'audio': 0}
    formatted_paths = []

    for path in media_paths:
        media_type = get_media_type(path)
        if media_type:
            media_counts[media_type] += 1

            # Multiple media files of the same type are not supported yet
            assert media_counts[
                media_type] <= 1, f'More than one {media_type} file found.'

            formatted_path = f'<{media_type}>{path}</{media_type}>'
            formatted_paths.append(formatted_path)

    return ''.join(formatted_paths)


def split_media(prompt: str):
    media_pattern = re.compile(r'<(image|audio|video)>(.+?)</\1>')
    parts = []
    last_end = 0
    for match in media_pattern.finditer(prompt):
        if last_end < match.start():
            text_segment = prompt[last_end:match.start()]
            parts.append(('text', text_segment))
        media_type = match.group(1)
        media_content = match.group(2)
        parts.append((media_type, media_content))
        last_end = match.end()
    if last_end < len(prompt):
        text_segment = prompt[last_end:]
        parts.append(('text', text_segment))

    return parts


def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def append_instructions(new_instructions, save_path):
    try:
        with open(save_path, 'r') as file:
            instructions = json.load(file)
    except FileNotFoundError:
        instructions = []

    instructions.extend(new_instructions)

    with open(save_path, 'w') as file:
        json.dump(instructions, file, ensure_ascii=False, indent=4)


def export_to_video(video_frames,
                    output_video_path: str = None,
                    fps: int = 10) -> str:
    # code from transformers.utils
    # use VP90 codec for browser compatibility

    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix='.mp4').name

    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8)
                        for frame in video_frames]

    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]

    fourcc = cv2.VideoWriter_fourcc(*'VP90')
    h, w, c = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path,
                                   fourcc,
                                   fps=fps,
                                   frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)
    return output_video_path
