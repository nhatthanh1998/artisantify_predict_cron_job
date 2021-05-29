import torch
from torchvision.transforms import transforms
from dotenv import load_dotenv
import os
import json
import boto3
import io


def init_s3_bucket(env, bucket):
    if env == "production":
        s3_client = boto3.client('s3')

    else:
        AWS_PUBLIC_KEY = os.environ.get("AWS_PUBLIC_KEY")
        AWS_PRIVATE_KEY = os.environ.get("AWS_PRIVATE_KEY")
        session = boto3.Session(
            aws_access_key_id=AWS_PUBLIC_KEY,
            aws_secret_access_key=AWS_PRIVATE_KEY
        )
        s3_client = session.client('s3')

    region = s3_client.get_bucket_location(Bucket=bucket)['LocationConstraint']

    return s3_client, region


load_dotenv()

ENV = os.environ.get("ENV", "dev")
S3_BUCKET = os.environ.get("S3_BUCKET")
s3, S3_REGION = init_s3_bucket(bucket=S3_BUCKET, env=ENV)


transform_tensor_to_pil_image = transforms.Compose(
    [
        transforms.ToPILImage()
    ]
)


def load_model(path, generator, device):
    generator.load_state_dict(torch.hub.load_state_dict_from_url(path, map_location=torch.device(device)))
    return generator


def transform():
    """ Transforms for training images """
    transform_ = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transform_


def transform_tensor_to_bytes(img):
    img = transform_tensor_to_pil_image(img)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()


def transform_byte_to_object(byte_data):
    response = byte_data.decode('utf8')
    response = json.loads(response)
    return response


def get_s3_location(key):
    return f'https://s3-{S3_REGION}.amazonaws.com/{S3_BUCKET}/{key}'


def save_image_to_s3(binary_data, key):
    s3.put_object(Body=binary_data, Bucket=S3_BUCKET, Key=key)
    return get_s3_location(key)
