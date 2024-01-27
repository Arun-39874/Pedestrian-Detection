from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
#net = cv2.dnn.readNet("sample/yolov3.weights", "sample/yolov3.cfg")
#net = cv2.dnn.readNet("sample/yolov3.weights", "sample/yolov3.cfg")

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f]

video_capture = cv2.VideoCapture("video.mp4")

@app.route('/')
def index():
    return render_template('index.html')

def generate():
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        height, width, _ = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)

        layer_names = net.getUnconnectedOutLayersNames()

        outs = net.forward(layer_names)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 0:
                    center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                    x, y = int(center_x - w / 2), int(center_y - h / 2)
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indices:
                box = boxes[i]
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host = "127.0.0.1", port = 8080, debug=True)
