import base64
import io
import json

import boto3
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

IMAGE_HEIGHT = 640
IMAGE_WIDTH = 640
COLOR = (255, 0, 0)
FONT_SIZE = 18

model = tf.keras.models.load_model("model")

font = ImageFont.truetype("Roboto-Bold.ttf", size=FONT_SIZE)

s3 = boto3.client("s3")


def load_image(bucket, key):
    return Image.open(s3.get_object(Bucket=bucket, Key=key)["Body"]).convert(
        "RGB"
    )


def detect(pil_image):
    image = tf.keras.preprocessing.image.img_to_array(pil_image)
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    image = image / 255
    boxes, scores, _, valid_detections = model(image[tf.newaxis])
    boxes = boxes[0].numpy()
    scores = scores[0].numpy()
    valid_detections = valid_detections[0].numpy()
    boxes = boxes[:valid_detections]
    scores = scores[:valid_detections]
    image_height, image_width = pil_image.size
    boxes = boxes * np.array(
        [image_width, image_height, image_width, image_height]
    )
    return boxes, scores


def annotate_image(pil_image, boxes, scores):
    draw = ImageDraw.Draw(pil_image)
    for (x1, y1, x2, y2), score in zip(boxes, scores):
        draw.rectangle((x1, y1, x2, y2), outline=COLOR, width=2)
        text = f"{score:.2f}"
        text_width, text_height = font.getsize(text)
        draw.rectangle((x1, y1, x1 + text_width, y1 - text_height), fill=COLOR)
        draw.text(
            (x1, y1 - text_height), text, fill=(255, 255, 255), font=font
        )
    return pil_image


def encode_image(pil_image):
    with io.BytesIO() as buffer:
        pil_image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode()


def get_response(pil_image, boxes, scores):
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(
            {
                "detections": [
                    {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": score}
                    for (x1, y1, x2, y2), score in zip(
                        boxes.tolist(), scores.tolist()
                    )
                ],
                "image": encode_image(pil_image),
            }
        ),
    }


def lambda_handler(event, context):
    bucket = event["queryStringParameters"]["bucket"]
    key = event["queryStringParameters"]["key"]

    pil_image = load_image(bucket, key)

    boxes, scores = detect(pil_image)

    annotate_image(pil_image, boxes, scores)

    return get_response(pil_image, boxes, scores)
