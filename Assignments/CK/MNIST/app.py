import base64
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load model đã train
model = tf.keras.models.load_model("model.h5", custom_objects={
                                   "softmax_v2": tf.nn.softmax})


def preprocess_image(image_base64):
    # Decode base64 thành ảnh
    image_data = base64.b64decode(image_base64.split(",")[1])
    image = Image.open(BytesIO(image_data)).convert("L")  # Chuyển về grayscale

    # Resize về kích thước 28x28 như MNIST
    image = image.resize((28, 28))
    image = np.array(image)

    # Chuẩn hóa dữ liệu giống như MNIST
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=[0, -1])  # Thêm batch dimension

    return image


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        image_base64 = data["image"]

        # Tiền xử lý ảnh
        processed_image = preprocess_image(image_base64)

        # Dự đoán chữ số
        prediction = model.predict(processed_image)
        predicted_label = int(np.argmax(prediction))

        return jsonify({"result": predicted_label})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
