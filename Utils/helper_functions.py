from PIL import Image
import io
import numpy as np
from flask import send_file
from keras.utils import img_to_array
import tensorflow as tf


def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((256, 256), Image.LANCZOS)
    img = img_to_array(img)
    img /= 255.0  # Normalize pixel values to [0, 1]
    tensor = tf.constant(img, dtype=np.float32)
    tensor = tf.expand_dims(tensor, axis=0)
    return tensor


def load_model(path, custom_objects):
    return tf.keras.models.load_model(path, custom_objects=custom_objects)

def tensor_to_image(tensor):
    # Convert the TensorFlow tensor to a PIL image
    tensor = tf.squeeze(tensor, axis=0)
    tensor = (tensor+1) * 127.5
    tensor = tensor.numpy().astype('uint8')
    image = Image.fromarray(tensor)
    return image

def serve_pil_image(pil_image):
    # Serve the PIL image as a response
    img_io = io.BytesIO()
    pil_image.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')
