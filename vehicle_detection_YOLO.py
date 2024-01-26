import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Load class names
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

# Read image
image = cv2.imread('sample_image.jpg')
height, width, _ = image.shape

# Preprocess image
blob = cv2.dnn.blobFromImage(image, scalefactor=1/255, size=(416, 416), swapRB=True, crop=False)

# Set input to YOLO model
net.setInput(blob)
output_layers_names = net.getUnconnectedOutLayersNames()
outs = net.forward(output_layers_names)

# Information to extract from detection
class_ids = []
confidences = []
boxes = []

# Extract information from output
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-maximum suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Count the detected vehicles
detected_vehicle_count = len(indexes)

# Draw bounding boxes
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (255, 0, 0)  # BGR format
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label + ' ' + str(round(confidence, 2)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the image with detected vehicles and count
cv2.putText(image, f"Detected Vehicles: {detected_vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.imshow('Vehicle Detection (YOLO)', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
