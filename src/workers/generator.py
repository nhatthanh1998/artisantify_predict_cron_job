from PIL import Image
from src.models.generator import GeneratorModel
import requests
from src.utils.utils import load_model, transform, save_generated_image
import uuid
import pika


class GeneratorWorker:
    def __init__(self, queue_name, queue_host, snapshot_path):
        self.snapshot_path = snapshot_path
        self.queue_name = queue_name
        self.queue_host = queue_host
        self.generator = GeneratorModel()
        self.transform_ = transform(image_size=256)
        self.generator = load_model(path=self.snapshot_path, generator=self.generator)

    def process_image(self, photoLocation, socketID, styleName):
        image_name = f"{uuid.uuid4()}.jpg"
        photo = Image.open(requests.get(photoLocation, stream=True).raw)
        photo = self.transform_(photo).unsqueeze(0)
        transform_image = self.generator(photo)
        save_generated_image(generated_image=transform_image, image_name=image_name)

    def start_task(self):
        connection = pika.BlockingConnection(pika.URLParameters(self.queue_host))
        channel = connection.channel()

        # create queue if it not exist
        channel.queue_declare(queue=self.queue_name, durable=True)

        channel.basic_consume(queue=self.queue_name, on_message_callback=self.process_image, auto_ack=True)

        print(' [*] Waiting for messages. To exit press CTRL+C')
        channel.start_consuming()