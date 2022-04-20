import numpy as np
from flask import Flask, request,  Response
import json
import cv2
import os
import matplotlib.pyplot as plt
import time

global Img

net = cv2.dnn.readNet(r"F:\FINAL year project\New folder\yolov4-tiny-custom_best.weights", r"F:\FINAL year project\New folder\yolov4-tiny-custom.cfg")
app = Flask(__name__)


def gen_frames():
    global Img  
    while True:
        if Img is None:
           continue

        frame = Img.copy()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
   return '<body> <div class="container"> \
   <div class="row"> <div class="col-lg-8  offset-lg-2"><h3 class="mt-5">Live Streaming</h3>\
   <img src="{{ url_for(\'video_feed\') }}" width="100%"> </div></div></div></body>'


@app.route('/upload', methods=['POST'])
def processImage():
    global Img
    if request.method == 'POST':
        nparr = np.frombuffer(request.files['imageFile'].read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        img = cv2.resize(img, (650 , 650))
        (height, width) = img.shape[:2]

        blob = cv2.dnn.blobFromImage(img, 1 / 255, (160, 160), (0, 0, 0), swapRB=True, crop=False)
        classes = ['Ball'] 
        
        boxes = []
        confidences = []
        class_ids = []

        net.setInput(blob)
        
        output_layers_name = net.getUnconnectedOutLayersNames()

        layerOutputs = net.forward(output_layers_name)  #layerOutputs

        result = {"x" : 0 , "y":0 , 'turn':""}

        for output in layerOutputs:
            for detection in output:
                score = detection[5:]
                class_id = np.argmax(score)
                confidence = score[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, .8, .4)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxes), 3))
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, "x = " + str(x + w // 2) + " " + "y =" + str(y + h // 2),
                            (x + w // 2 - 100, y + h + 50), font, 1.75, color, 2)
                cv2.putText(img, label + " " + confidence, (x + w // 2 - 80, y + h + 100), font, 1.5, color, 2)
            
            Img = img.copy()
           
            result['x'] = x + w//2
            result['y'] = y + h//2
            cv2.imwrite("ouput.jpg", img)
        else:
            cv2.imwrite("ouput.jpg", img)

        #Object Tracking Conditions
        if(x>350):
            result['turn'] = "Right"
        elif(x>300 and x<350):
            if(y<300):
                result['turn'] = "Forward"
            elif(y>300 and y<350):
                result['turn'] = "Stop"
            else:
                result['turn'] = "Backward"
        else:
            result['turn'] = "Left"
          
        return json.dumps(result)

    return "falied"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500 , debug = True)