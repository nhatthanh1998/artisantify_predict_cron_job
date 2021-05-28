import torch.cuda
from PIL import Image
from src.models.generator import Generator
import requests
from src.utils.utils import load_model, transform, save_generated_image, transform_byte_to_object
import uuid
import pika


class GeneratorWorker:
    def __init__(self, queue_host, exchange_name, snapshot_path, main_server_endpoint):
        self.snapshot_path = snapshot_path
        self.queue_host = queue_host
        self.exchange_name = exchange_name
        self.main_server_endpoint = main_server_endpoint
        self.generator = Generator()
        self.transform_ = transform()
        self.generator = load_model(path=self.snapshot_path, generator=self.generator)

    def process_image(self, ch, method, properties, body):
        print("message coming!!!!!")
        body = transform_byte_to_object(body)
        # extract data from body
        data = body['data']
        socketId = data['socketId']
        accessURL = data['accessURL']
        image_name = f"{uuid.uuid4()}.jpg"
        photo = Image.open(requests.get(accessURL, stream=True).raw)
        photo = self.transform_(photo).unsqueeze(0)
        print(photo.shape)
        transform_image = self.generator(photo)
        save_generated_image(generated_image=transform_image, image_name=image_name)
        endpoint_url = f"{self.main_server_endpoint}/photos/transfer-photo/completed"
        data = {'socketId': socketId, 'transferPhotoName': f'images/{image_name}'}
        requests.post(endpoint_url, data=data)
        torch.cuda.empty_cache()

    def start_task(self):
        connection = pika.BlockingConnection(pika.URLParameters(self.queue_host))
        channel = connection.channel()
        # create queue if it not exist
        channel.exchange_declare(exchange=self.exchange_name, exchange_type='direct')
        result = channel.queue_declare(queue='', exclusive=True)
        queue_name = result.method.queue
        channel.queue_bind(exchange=self.exchange_name, queue=queue_name)
        channel.basic_consume(queue=queue_name, on_message_callback=self.process_image, auto_ack=True)

        print(' [*] Waiting for messages. To exit press CTRL+C')
        channel.start_consuming()
