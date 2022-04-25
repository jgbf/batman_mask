import cv2 as cv


class FaceFinder:
    def __init__(self, algorithm):
        self.cascade = cv.CascadeClassifier(algorithm)

    def find_faces(self, frame, threshold=80):
        """
        Function to detect faces on an openCV frame 

        :param frame: openCV image frame 
        :param threshold: minimum size of face what we will detect on image
        """
        
        # Convert frame to grayscale and detect faces on it
        gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray_image, 1.1, 4)

        # Filter faces based on the given threshold
        filtered_faces = [element for element in faces if element[3] > threshold]
        
        return filtered_faces
