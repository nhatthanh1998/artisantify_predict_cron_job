import pika
import sys
import os
from dotenv import load_dotenv
from workers.generator import GeneratorWorker
import argparse
import requests


load_dotenv()
QUEUE_NAME = os.environ.get("QUEUE_NAME")
QUEUE_HOST = os.environ.get("QUEUE_HOST")
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
            generator_worker = GeneratorWorker(queue_name=QUEUE_NAME, queue_host=QUEUE_HOST,
                                               snapshot_path='model_weights/mosaic/1',
                                               main_server_endpoint=MAIN_SERVER_ENDPOINT
                                            )
            generator_worker.start_task()
        except KeyboardInterrupt:
            print('Interrupted')
            try:
                sys.exit(0)
            except SystemExit:
                sys.exit(0)
