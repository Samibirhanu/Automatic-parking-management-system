from flask import Flask, render_template, request
import os
from yolo_deep_learning_integartion_copy import yolo_OCR
from color_detection import plate_color_detection
# webserver gateway interface
app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, './static/upload/')

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        upload_file = request.files['image_name']
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH, filename)
        upload_file.save(path_save)
        text = yolo_OCR(path_save, filename)
        color = plate_color_detection(path_save)
        return render_template('index.html', upload=True, upload_image = filename, text = text , color = color)


    return render_template('index.html', upload=False)

if __name__ =="__main__":
    app.run(debug=True)

