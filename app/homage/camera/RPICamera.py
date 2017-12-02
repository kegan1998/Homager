

from Camera import Camera
import time
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
from elk import ElkAttribute


class RPICamera(Camera):
    stream = ElkAttribute(mode='rw',
        builder='_stream', lazy=True)

    def _camera(self):
        try:
            #from picamera.array import PiRGBArray
            #from picamera import PiCamera
            pass
        except ImportError:
            print("PI camera is not detected")
            return None

        camera = PiCamera()
        camera.resolution = (640, 480)
        #camera.framerate = 32
        return camera

    def _stream(self):
        stream = PiRGBArray(self.camera, size=(640, 480))
        # allow the camera to warmup
        time.sleep(1)
        return stream

    def capture(self):
        self.stream.truncate(0)
        self.camera.capture(self.stream,
             format="bgr", use_video_port=True)
        # grab the raw NumPy array representing the image,
        # then initialize the timestamp
        # and occupied/unoccupied text
        image = self.stream.array
        return image

if __name__ == "__main__":
    c = RPICamera()
    while(True):
        image = c.capture()
        #grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # where should it be?
        #c.rawCapture.truncate(0)
    cv2.destroyAllWindows()
