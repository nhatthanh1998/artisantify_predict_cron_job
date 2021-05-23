import torch
from torchvision.transforms import transforms


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


def save_image():
    pass