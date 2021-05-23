import pika
import sys
import os
from dotenv import load_dotenv
from src.tasks.queue_consumer_task import start_task


load_dotenv()
QUEUE_NAME = os.environ.get("QUEUE_NAME")
QUEUE_HOST = os.environ.get("QUEUE_HOST")

if __name__ == '__main__':
    try:
        start_task(queue_name=QUEUE_NAME, queue_host=QUEUE_HOST)
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os.exit(0)
