import torch
from torchvision.transforms import transforms
from torchvision.utils import save_image
from dotenv import load_dotenv
import os
import json
from smart_open import open
load_dotenv()


def load_model(path, generator):
    generator.load_state_dict(torch.hub.load_state_dict_from_url(path))
    return generator


def transform():
    """ Transforms for training images """
    transform_ = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transform_


def save_generated_image(generated_image, image_name):
    PUBLIC_DIR = os.environ.get("PUBLIC_DIR")
    save_dir = os.path.abspath(PUBLIC_DIR)
    SAVE_DIR = os.path.join(save_dir, image_name)
    save_image(generated_image, SAVE_DIR)


def transform_byte_to_object(byte_data):
    response = byte_data.decode('utf8')
    response = json.loads(response)
    return response
