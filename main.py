import numpy as np
import cv2 as cv
from datetime import datetime

from facefinder import FaceFinder
from masker import Masker


def main():
    # Create face detector object
    face_finder = FaceFinder('haarcascade_frontalface_default.xml')
    
    # Create mask applyer object
    masker = Masker('batman_small.png')

    # Initialize webcam video capture 
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Find faces on the givne frame 
        faces = face_finder.find_faces(frame)

        # Add masks to the frame where we found the faces
        frame = masker.add_mask(frame, faces)
        
        # Display the resulting frame
        cv.imshow('Filter Camera', frame)

        # Handle button presses
        if cv.waitKey(1) == ord('q'):
            break

        if cv.waitKey(1) == ord('c'):
            cv.imwrite(f'{str(datetime.now().date())}_screenshot.png', frame)
            print('Image saved')

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
