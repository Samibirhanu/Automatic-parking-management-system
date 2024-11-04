from flask import Flask, render_template, request
import os
from yolo_deep_learning_integartion_copy import yolo_OCR, plate_number_sequence
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
        text, coords = yolo_OCR(path_save, filename)
        extracted_plate_number = plate_number_sequence(text)
        color = plate_color_detection(path_save, coords)
        return render_template('index2.html', upload=True, upload_image = filename, text = text , color = color , extracted_plate_number = extracted_plate_number)


    return render_template('index2.html', upload=False)

if __name__ =="__main__":
    app.run(debug=True)

