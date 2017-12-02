
from app.homage.camera.Camera import Camera
from elk import Elk, ElkAttribute

class TestCamera(Camera):
    image = ElkAttribute(mode='rw')

    def _camera(self):
        return None

    def capture(self):
        return self.image
