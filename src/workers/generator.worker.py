from PIL import Image
from src.models.generator import GeneratorModel
import requests
from src.utils.utils import load_model, transform
from torchvision import transforms


class GeneratorWorker:
    def __init__(self):
        self.generator = GeneratorModel()
        self.transform_ = transform()
        self.generator = load_model(path='src/models/starry_night/1', generator=self.generator)

    def process_image(self, photo_location):
        photo = Image.open(requests.get(photo_location, stream=True).raw)
        photo = self.transform_(photo).unsqueeze(0)
        transform_image = self.generator(photo)
        image = transforms.ToPILImage()(transform_image[0]).convert("RGB")
