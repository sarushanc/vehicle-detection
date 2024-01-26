import cv2 # pip install opencv-python
import numpy as np # pip install numpy
import tkinter as tk # pip install tkinter
from PIL import Image, ImageTk # pip install pillow

# Load YOLO model and classes
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

# Create GUI window
root = tk.Tk()
root.title("Vehicle Detection")

# Create canvas to display image
canvas = tk.Canvas(root, width=800, height=600)
canvas.pack(side=tk.LEFT, padx=10, pady=10)

# Create frame for vehicle details
details_frame = tk.Frame(root)
details_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

# Create scrollbar for details
scrollbar = tk.Scrollbar(details_frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Create listbox for vehicle details
details_listbox = tk.Listbox(details_frame, yscrollcommand=scrollbar.set)
details_listbox.pack(fill=tk.BOTH, expand=True)
scrollbar.config(command=details_listbox.yview)

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
vehicle_counts = {}

# Display the image with detected vehicles and confidence
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        if label not in vehicle_counts:
            vehicle_counts[label] = 1
        else:
            vehicle_counts[label] += 1
        color = (255, 0, 0)  # BGR format
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label + ' ' + str(round(confidence, 2)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the image with detected vehicles and details
total_vehicle_count = 0
for label, count in vehicle_counts.items():
    details_listbox.insert(tk.END, f"{label}: {count}")
    total_vehicle_count += count

# Display total vehicle count
details_listbox.insert(tk.END, f"Total Vehicles: {total_vehicle_count}")

# Display the image with detected vehicles
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_pil = Image.fromarray(image)
image_tk = ImageTk.PhotoImage(image_pil)
canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)

# Start the GUI event loop
root.mainloop()
