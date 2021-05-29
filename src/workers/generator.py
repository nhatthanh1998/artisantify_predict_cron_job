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


    def preprocess(self, photo_access_url):
        model_input = Image.open(requests.get(photo_access_url, stream=True).raw)
        model_input = self.transform_(model_input).unsqueeze(0)
        return model_input


    def inference(self, model_input):
        return self.generator(model_input)


    def postprocess(self, model_output, image_name, socketId):
        byte_data = transform_tensor_to_bytes(model_output)
        image_location = save_image_to_s3(byte_data, image_name)

        endpoint_url = f"{self.main_server_endpoint}/photos/transfer-photo/completed"

        payload = {'socketId': socketId, 'transferPhotoLocation': image_location}    
        requests.post(endpoint_url, data=payload)
        torch.cuda.empty_cache()


    def handler(self, ch, method, photo_access_url, socketId, image_name):
        #1. Preprocess
        model_input = self.preprocess(photo_access_url=photo_access_url)

        #2. Transform
        model_output = self.inference(model_input=model_input)

        #3. Postprocess
        self.postprocess(model_output=model_output, image_name=image_name, socketId=socketId)

        #4. Ack the processed message.
        ch.basic_ack(delivery_tag = method.delivery_tag)        


    def process_message(self, ch, method, properties, body):
        print("New message coming!!!!!")
        body = transform_byte_to_object(body)
        # extract data from body
        data = body['data']
        socketId = data['socketId']
        accessURL = data['accessURL']
        date_time = datetime.now().strftime("%m-%d-%Y")
        image_name = f"{date_time}/{uuid.uuid4()}.jpg"

        # Put data to model process pipeline
        self.handler(ch=ch, method=method, photo_access_url=accessURL, socketId=socketId, image_name=image_name)


    def start_task(self):
        connection = pika.BlockingConnection(pika.URLParameters(self.queue_host))
        channel = connection.channel()
        # create queue if it not exist
        channel.exchange_declare(exchange=self.exchange_name, exchange_type='direct', durable=True)
        channel.queue_bind(exchange=self.exchange_name, queue = self.queue_name)
        channel.basic_consume(queue=self.queue_name, on_message_callback=self.process_message)

        print(f' [*] Waiting for messages at exchange {self.exchange_name}.  To exit press CTRL+C')
        channel.start_consuming()
