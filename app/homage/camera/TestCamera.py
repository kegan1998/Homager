
from app.homage.camera.Camera import Camera
from elk import ElkAttribute

class TestCamera(Camera):
    image = ElkAttribute(mode='rw', required=True)

    def _camera(self):
        return None

    def capture(self):
        return self.image
