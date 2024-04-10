from flask import Flask 
from PIL import Image
from flask import request
import base64
import io
import numpy as np
from image_object_detection import yolo_object_detection
from markupsafe import escape

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>HELLO WORLD!<p>"

@app.route("/image", methods = ['POST'])
def index_page():
    img = request.form['image']

    img_bytes = base64.b64decode(img)

    image = Image.open(io.BytesIO(img_bytes))

    # Convert PIL Image to numpy array
    img_np = np.array(image)
    img = img_np

    # im_arr = np.frombuffer(img_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    # img = cv.imdecode(im_arr, flags=cv.IMREAD_COLOR)

    class_names = yolo_object_detection(img)

    return str(class_names)

if __name__ == '__main__':
    app.run(debug=True)
