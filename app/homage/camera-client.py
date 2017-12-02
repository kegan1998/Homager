
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
from FaceAgent import FaceAgent

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

class RestRecognition(elk.Elk):
    rest = elk.ElkAttribute(mode='ro', type=Rest,
        builder="_rest") 
    base_url = elk.ElkAttribute(mode='ro', type=str,
        default="http://127.0.0.1:5000/faces/api/v1.0")
    video_capture = elk.ElkAttribute(mode='ro',
        builder="_video_capture", lazy=True)
    face_agent = elk.ElkAttribute(mode='ro', type=FaceAgent,
        builder = '_face_agent', lazy=1) 

    def _face_agent(self):
        return FaceAgent()

    def __del__(self):
        self.video_capture.release()

    def _video_capture(self):
        return cv2.VideoCapture(0)

    def _rest(self):
        return Rest(base_url=self.base_url) 

    def recognize(self,what,image):
        retval, buffer = cv2.imencode('.png', image)
        image_as_text = base64.b64encode(buffer)
        response = self.rest.post("/"+what, {what:image_as_text})
        return response

    def destroy(self):
        self.video_capture.release();
#        requests.release()
    
    def run_full(self):
        log.info("running live recognition")

#         try:
#             image = cv2.imread('input.png', cv2.COLOR_BGR2GRAY)
#             faces = self.recognize_image(image)
#             if faces:
#                 print(faces)
#     	except:
#     	    log.error( "Failed to recognize faces" )
#             finally:
#                 self.destroy()
#                 sys.exit(1)
#        cv.imread()

        font = cv2.FONT_HERSHEY_SIMPLEX
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                 raise "Failed to capture image from camera"
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            response = self.recognize(image)
            #response = {u'faces':[]}
            for face in response[u'faces']:
                location = face[u'location']
                size = face[u'size']
                cv2.rectangle(image, location, (location[0]+size[0], location[1]+size[1]), (0, 255, 0), 2)
                if face[u'confidence'] > 150:
                    #self.new_face(face['image'])
                    pass
                else:
                    cv2.putText(
                        image,
                        self.decode_face_id(face[u'predicted'])+':'+str(face[u'confidence']),
                        (location[0],location[1]+size[1]),
                        font,
                        1,
                        (255,255,255),
                        2,
                        cv2.CV_AA)
            cv2.imshow('Video', image)
            cv2.waitKey(200)

    def run_face(self):
        log.info("running live recognition")
        font = cv2.FONT_HERSHEY_SIMPLEX
        agent = self.face_agent
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                 raise "Failed to capture image from camera"
            image = frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for (x, y, w, h) in agent.detect_faces(image):
                face_image = image[y: y + h, x: x + w]
                face_image = agent.normalize_face_image(face_image)
                log.info("face is detected at "+str(x)+","+str(y))
                response = self.recognize(u'face',face_image)
                predicted = response[u'predicted']
                confidence = response[u'confidence']
                log.info("face is recognized as '" + predicted + "' with confidence " + str(confidence))
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                #cv2.rectangle(image, location, (location[0]+size[0], location[1]+size[1]), (0, 255, 0), 2)
                if confidence > 250:
                    #self.new_face(face['image'])
                    pass
                else:
                    cv2.putText(
                        image,
                        predicted+':'+str(confidence),
                        (x,y+h),
                        font,
                        1,
                        (255,255,255),
                        2,
                        cv2.CV_AA)
            cv2.imshow('Video', image)
            cv2.waitKey(200)

if __name__ == "__main__":
    log.basicConfig(level=log.INFO)
    r = RestRecognition()
    r.run_face()
