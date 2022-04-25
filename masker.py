import cv2 as cv


class Masker():

    def __init__(self, mask_path):
        self.mask = cv.imread(mask_path, -1)
        self.ratio = self.mask.shape[0] / self.mask.shape[1]

    def add_mask(self, frame, faces):
        """
        Function to add pre-defined masks to the frame

        :param frame: OpenCV frame 
        :param faces: list of faces freated by a the FaceFinder object
        """

        for (x_offset, y_offset, width, height) in faces:
            # set default values to prevent false detections
            x_offset = max(x_offset, 1)
            y_offset = max(y_offset, 1)
            width = max(width, 1)
            height = max(height, 1)

            # Set mask size
            dims = (int((width / self.ratio) * 1.4), int(height * 1.4))

            if dims[0] == 0:
                dims = (1,1)
            
            # Resize image
            resized_mask = cv.resize(self.mask, dims, interpolation = cv.INTER_AREA)

            
            # Move mask to the face closer to center
            x_offset += width // 10
            y_offset -= height // 2

            # Calcualte place on frame
            y1, y2 = y_offset, y_offset + resized_mask.shape[0]
            x1, x2 = x_offset, x_offset + resized_mask.shape[1]

            alpha_s = resized_mask[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            # Add mask to image
            if x1 > 0 and y1 > 0 and x2 > 0 and y2 >0:
                for c in range(0, 3):
                    batman_alpha = alpha_s * resized_mask[:, :, c]
                    frame_alpha = alpha_l * frame[y1:y2, x1:x2, c]
                    frame[y1:y2, x1:x2, c] = batman_alpha + frame_alpha

        return frame
