from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from ai_utils.utils import decodeImage
from research.obj import MultiClassObj


app = Flask(__name__)
CORS(app)


#@cross_origin()
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        modelPath = 'research/ssd_mobilenet_v1_coco_2017_11_17'
        self.objectDetection = MultiClassObj(self.filename, modelPath)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.objectDetection.getPrediction()
    return jsonify(result)


#port = int(os.getenv("PORT"))
if __name__ == "__main__":
    clApp = ClientApp()
    #app.run(host='0.0.0.0', port=port)
    app.run(host='127.0.0.1', port=7500, debug=True)
