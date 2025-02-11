# Computer Vision
# Laboratory 1.6
# 
# Authors:
# - Diego Estuardo Lemus Lopez - 21469
# - José Pablo Kiesling Lange - 21581
# - Herber Sebastián Silva Munioz - 21764
# 
# Date: february 2025
# Usage: python3 lab1_6.py
# Dependencies: numpy, opencv-python

import cv2
import numpy as np

# Predefined HSV color ranges
COLOR_HSV = {
    "YELLOW": ([20, 100, 100], [30, 255, 255]),
    "RED": ([0, 100, 100], [10, 255, 255]),
    "RED_ALT": ([170, 100, 100], [180, 255, 255]),
    "BLUE": ([100, 100, 100], [140, 255, 255]),
    "GREEN": ([40, 100, 100], [80, 255, 255])
}

CURRENT_COLOR = "GREEN" 

"""
    Function to detect objects of a specific color using a webcam.
    The function reads the frames from the webcam, converts them to the HSV color space, 
    and applies a color segmentation mask.

    The function displays two windows:
    - Color Detector: Displays the original frame with the bounding boxes of the detected objects
    - Mask: Displays the mask used to segment the objects
"""
def detector():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to the HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get the HSV color range for the selected color
        if CURRENT_COLOR == "RED": # Red is a special case
            low_range1, high_range1 = COLOR_HSV["RED"]
            low_range2, high_range2 = COLOR_HSV["RED_ALT"]
            
            mask1 = cv2.inRange(hsv, np.array(low_range1), np.array(high_range1))
            mask2 = cv2.inRange(hsv, np.array(low_range2), np.array(high_range2))
            mascara = cv2.bitwise_or(mask1, mask2)
            
        else: # Other colors
            low_range, high_range = COLOR_HSV.get(CURRENT_COLOR, ([0, 0, 0], [0, 0, 0]))
            mascara = cv2.inRange(hsv, np.array(low_range), np.array(high_range))
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
        
        # Find the contours in the mask
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Define the bounding box for each contour
        for contorno in contornos:
            if cv2.contourArea(contorno) > 500:
                x, y, w, h = cv2.boundingRect(contorno)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, CURRENT_COLOR.capitalize(), (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show the frames
        cv2.imshow("Color Detector", frame)
        cv2.imshow("Mask", mascara)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detector()
