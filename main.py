import cv2  # Import the OpenCV library for computer vision tasks
import pandas as pd  # Import Pandas for data manipulation and analysis
from ultralytics import YOLO  # Import the YOLO model from the ultralytics package for object detection
import numpy as np  # Import NumPy for numerical operations and handling arrays
import pytesseract  # Import Pytesseract for optical character recognition (OCR)
from datetime import datetime  # Import datetime for handling date and time

# Optical Character Recognition setup
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Set the path for the Tesseract executable
model = YOLO('best.pt')  # Load the YOLO model with the specified weights

# Function to get mouse coordinates
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  # Check if the mouse is moved
        point = [x, y]  # Store the x and y coordinates of the mouse
        print(point)  # Print the coordinates

# Create a window named 'RGB' and set a mouse callback function
cv2.namedWindow('RGB')  
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('video\mycarplate.mp4')  # Open the video file for capturing frames

# Read the class labels from the 'coco.txt' file
my_file = open('coco.txt', 'r')  # Open the class names file in read mode
data = my_file.read()  # Read the content of the file
class_list = data.split("\n")  # Split the content into a list of class names

# Define the area of interest for the license plate detection
area = [(27, 417), (16, 456), (1015, 451), (992, 417)]

count = 0  # Initialize a frame counter
list1 = []  # List to store recognized license plate numbers
processed_numbers = set()  # Set to keep track of processed license plate numbers

# Open a file to write car plate data
with open("car_plate_data.txt", "a") as file:
    file.write("NumberPlate\tDate\tTime\n")  # Write column headers for the data

while True:    
    ret, frame = cap.read()  # Read a frame from the video
    count += 1  # Increment the frame counter
    if count % 3 != 0:  # Skip every 3rd frame
        continue
    if not ret:  # If there are no frames left to read, exit the loop
       break
   
    frame = cv2.resize(frame, (1020, 500))  # Resize the frame for consistent processing
    results = model.predict(frame)  # Use the YOLO model to detect objects in the frame
    a = results[0].boxes.data  # Get the bounding box data from the model results
    px = pd.DataFrame(a).astype("float")  # Convert the bounding box data to a DataFrame and ensure float type
   
    for index, row in px.iterrows():  # Iterate over each detected object
        x1 = int(row[0])  # Get the x1 coordinate of the bounding box
        y1 = int(row[1])  # Get the y1 coordinate of the bounding box
        x2 = int(row[2])  # Get the x2 coordinate of the bounding box
        y2 = int(row[3])  # Get the y2 coordinate of the bounding box
        
        d = int(row[5])  # Get the class index of the detected object
        c = class_list[d]  # Get the class name using the index
        cx = int(x1 + x2) // 2  # Calculate the center x coordinate of the bounding box
        cy = int(y1 + y2) // 2  # Calculate the center y coordinate of the bounding box
        result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)  # Check if the center is within the defined area
        if result >= 0:  # If the center point is within the area
            crop = frame[y1:y2, x1:x2]  # Crop the detected license plate region
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)  # Convert the cropped image to grayscale
            gray = cv2.bilateralFilter(gray, 10, 20, 20)  # Apply bilateral filter for noise reduction
            print(type(gray), gray.shape)  # Print the type and shape of the processed image

            text = pytesseract.image_to_string(gray).strip()  # Use Pytesseract to extract text from the image
            text = text.replace('(', '').replace(')', '').replace(',', '').replace(']', '').replace('[', '')  # Clean up the extracted text
            print(text)  # Print the recognized text
            if text not in processed_numbers:  # If the text has not been processed yet
                processed_numbers.add(text)  # Add the text to the set of processed numbers
                list1.append(text)  # Append the recognized license plate to the list
                current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get the current date and time
                with open("car_plate_data.txt", "a") as file:  # Open the data file in append mode
                    file.write(f"{text}\t{current_datetime}\n")  # Write the license plate and timestamp to the file
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw a rectangle around the detected license plate
            cv2.imshow('crop', crop)  # Show the cropped image of the license plate
      
    cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 0), 2)  # Draw the defined area of interest on the frame
    cv2.imshow("RGB", frame)  # Display the current frame
    cv2.waitKey(1)  # Wait for 1 millisecond for a key press

cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows
