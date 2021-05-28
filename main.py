import pika
import sys
import os
from dotenv import load_dotenv
from src.workers.generator import GeneratorWorker
import argparse
import requests
import json


load_dotenv()
EXCHANGE_NAME = os.environ.get("EXCHANGE_NAME")
QUEUE_HOST = os.environ.get("QUEUE_HOST")
QUEUE_NAME = os.environ.get("QUEUE_NAME")

MAIN_SERVER_ENDPOINT = os.environ.get("MAIN_SERVER_ENDPOINT")
parser = argparse.ArgumentParser()

parser.add_argument("--styleID", type=str, help="styleID that the model belong to", default='', required=True)
params = parser.parse_args()
styleID = params.styleID


if len(styleID) == 0:
    raise ValueError("StyleID must be filled!!!")
else:
    if __name__ == '__main__':
        try:
            response = requests.get(f"{MAIN_SERVER_ENDPOINT}/styles/{styleID}/active-model")
            data = json.loads(response.content.decode('utf-8'))
            
            exchangeName = data.get("exchangeName")
            modelType = data.get("modelType")
            snapshotPath = data.get("snapshotPath")
            generator_worker = GeneratorWorker(exchange_name=exchangeName, queue_host=QUEUE_HOST,
                                               queue_name = QUEUE_NAME,
                                               snapshot_path=snapshotPath,
                                               main_server_endpoint=MAIN_SERVER_ENDPOINT
                                               )
            generator_worker.start_task()
        except KeyboardInterrupt:
            print('Interrupted')
            try:
                sys.exit(0)
            except SystemExit:
                sys.exit(0)
