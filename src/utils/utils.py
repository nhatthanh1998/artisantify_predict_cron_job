import torch
from torchvision.transforms import transforms
from torchvision.utils import save_image
from dotenv import load_dotenv
import os


load_dotenv()


def load_model(path, generator):
    generator.load_state_dict(torch.load(path + '/generator.pth'))
    return generator


def transform(image_size):
    """ Transforms for training images """
    transform_ = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transform_


def save_generated_image(generated_image, image_name):
    PUBLIC_DIR = os.environ.get("PUBLIC_DIR")
    SAVE_DIR = f"{PUBLIC_DIR}/images/{image_name}"
    save_image(generated_image, SAVE_DIR)
