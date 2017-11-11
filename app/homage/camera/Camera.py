
import cv2
import os
import elk

class Camera:
    camera = elk.ElkAttribute(mode='ro',
        builder='_camera', lazy=True)

    def __init__(self):
        self.camera = None

    def __del__(self):
        if self.camera is not None:
            self.camera.release()

    def _camera(self):
        return cv2.VideoCapture(0)
    
    def capture(self):
        ret, frame = self.camera.read()
        return frame

if __name__ == "__main__":
    c = Camera()
    while(True):
        image = c.capture()
        #grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame',image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
