
from app.homage.Camera import Camera
import time
import cv2
import elk

#import os
# os.uname() eq 'armv6l' 

class RPICamera(Camera):
    def _camera(self):
        try:
            from picamera.array import PiRGBArray
            from picamera import PiCamera
        except ImportError, e:
            return None

        camera = PiCamera()
        camera.resolution = (640, 480)
        camera.framerate = 32
        self.rawCapture = PiRGBArray(camera, size=(640, 480))
        # allow the camera to warmup
        time.sleep(0.1)
        return camera             
   
    def capture(self):
        frame = self.camera.capture_continuous(self.rawCapture,
             format="bgr", use_video_port=True)
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        image = frame.array
        return image

if __name__ == "__main__":
    c = RPICamera()
    while(True):
        image = c.capture()
        #grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame',image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # where should it be?
        c.rawCapture.truncate(0)
    cv2.destroyAllWindows()
