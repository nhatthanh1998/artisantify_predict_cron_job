import torch
from PIL import Image
from src.models.generator import Generator
import requests
from src.utils.utils import load_model, transform, transform_byte_to_object, save_image_to_s3, transform_tensor_to_bytes
import uuid
import pika
from datetime import datetime


class GeneratorWorker:
    def __init__(self, queue_host, queue_name, exchange_name, snapshot_path, main_server_endpoint):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.snapshot_path = snapshot_path
        self.queue_host = queue_host
        self.queue_name = queue_name
        self.exchange_name = exchange_name
        self.main_server_endpoint = main_server_endpoint
        self.generator = Generator().to(self.device)
        self.transform_ = transform()
        self.generator = load_model(path=self.snapshot_path, generator=self.generator, device=self.device)

    def process_image(self, ch, method, properties, body):
        print("message coming!!!!!")
        body = transform_byte_to_object(body)
        # extract data from body
        data = body['data']
        socketId = data['socketId']
        accessURL = data['accessURL']
        date_time = datetime.now().strftime("%m-%d-%Y")
        image_name = f"{date_time}/{uuid.uuid4()}.jpg"
        photo = Image.open(requests.get(accessURL, stream=True).raw)
        photo = self.transform_(photo).unsqueeze(0)
        print(photo.shape)
        transform_image = self.generator(photo)
        transform_image = transform_tensor_to_bytes(transform_image)
        image_location = save_image_to_s3(transform_image, image_name)
        endpoint_url = f"{self.main_server_endpoint}/photos/transfer-photo/completed"
        data = {'socketId': socketId, 'transferPhotoLocation': image_location}
        print(data)
        requests.post(endpoint_url, data=data)
        torch.cuda.empty_cache()

    def start_task(self):
        connection = pika.BlockingConnection(pika.URLParameters(self.queue_host))
        channel = connection.channel()
        # create queue if it not exist
        channel.exchange_declare(exchange='style-name-1', exchange_type='direct')
        channel.queue_bind(exchange='style-name-1', queue = self.queue_name)
        channel.basic_consume(queue=self.queue_name, on_message_callback=self.process_image, auto_ack=True)

        print(f' [*] Waiting for messages at exchange {self.exchange_name}.  To exit press CTRL+C')
        channel.start_consuming()
