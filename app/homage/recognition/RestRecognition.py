
import logging as log
import urllib2
import requests
import json
import weakref
import base64
#from restful_lib import Connection
import elk
import cv2
import sys
from app.homage.FaceAgent import FaceAgent
from app.homage.recognition.Recognition import Recognition

class Rest(elk.Elk):
    username = elk.ElkAttribute(mode='rw', type=str,
        default="admin") 
    password = elk.ElkAttribute(mode='rw', type=str,
        default="admin") 
    base_url = elk.ElkAttribute(mode='ro', type=str,
        default="http://127.0.0.1:5000/faces/api/v1.0")
    def get(self,path):
        response = requests.get(self.base_url + path,
            auth=(self.username, self.password))
        data = response.json()
        return data

    def post(self,path,data):
        response = requests.post(self.base_url + path,
            auth=(self.username, self.password),
            json=json.dumps(data) )
        if not response.ok:
            response.raise_for_status()
        data = response.json()
        return data

class RestRecognition(Recognition):
    rest = elk.ElkAttribute(mode='ro', type=Rest,
        builder="_rest")
    address = elk.ElkAttribute(mode='ro', type=str,
        default="127.0.0.1")
    base_url = elk.ElkAttribute(mode='ro', type=str,
        builder='_base_url', lazy=True)

    def _base_url(self):
        return "http://{}:5000/faces/api/v1.0" % (self.address)

    def _rest(self):
        return Rest(base_url=self.base_url) 

    def recognize(self,what,image):
        retval, buffer = cv2.imencode('.png', image)
        image_as_text = base64.b64encode(buffer)
        response = self.rest.post("/"+what, {what:image_as_text})
        return response

if __name__ == "__main__":
    log.basicConfig(level=log.INFO)
    r = RestRecognition()
#             image = cv2.imread('input.png', cv2.COLOR_BGR2GRAY)
#             faces = self.recognize_image(image)
