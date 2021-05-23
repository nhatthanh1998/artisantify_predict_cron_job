from PIL import Image
from src.models.generator import GeneratorModel
import requests
from src.utils.utils import load_model, transform, save_generated_image
import uuid

class GeneratorWorker:
    def __init__(self):
        self.generator = GeneratorModel()
        self.transform_ = transform()
        self.generator = load_model(path='src/models/starry_night/1', generator=self.generator)

    def process_image(self, photoLocation, socketID, styleName):
        image_name = f"{uuid.uuid4()}.jpg"
        photo = Image.open(requests.get(photoLocation, stream=True).raw)
        photo = self.transform_(photo).unsqueeze(0)
        transform_image = self.generator(photo)
        save_generated_image(generated_image=transform_image, image_name=image_name)
