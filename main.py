import pika
import sys
import os
from dotenv import load_dotenv
from src.workers.generator import GeneratorWorker

load_dotenv()
QUEUE_NAME = os.environ.get("QUEUE_NAME")
QUEUE_HOST = os.environ.get("QUEUE_HOST")

if __name__ == '__main__':
    try:
        generator_worker = GeneratorWorker(queue_name=QUEUE_NAME, queue_host=QUEUE_HOST,
                                           snapshot_path='src/model_weights/mosaic/1')
        generator_worker.start_task()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os.exit(0)
