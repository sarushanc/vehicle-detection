import cv2

# Load the pre-trained Haar Cascade classifier for vehicle detection
vehicle_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

# Read an image (you can replace this with your own image)
image = cv2.imread('20211214-bev-07.webp')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect vehicles in the image
vehicles = vehicle_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around the detected vehicles and count them
vehicle_count = 0
for (x, y, w, h) in vehicles:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    vehicle_count += 1

# Display the image with detected vehicles and the count
cv2.putText(image, f'Total Vehicles: {vehicle_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imshow('Vehicle Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
