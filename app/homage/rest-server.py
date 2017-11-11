from flask import Flask, jsonify, request, abort, g
import logging as log
import json
import base64
import cv2
import numpy as np
from FaceAgent import FaceAgent 

images = [
    {
        'id': 1,
        'data' : None,
    },
]
log.basicConfig(level=log.INFO)
face_agent = FaceAgent()
face_agent.init()
#face_agent = get_face_agent()
app = Flask(__name__)

def get_face_agent():
    #return face_agent
    #agent = g.get('agent', None)
    if face_agent is None:
        face_agent = FaceAgent()
        log.info("learning agent")
        face_agent.init()
        #g.agent = agent
    return face_agent

@app.route('/faces/api/v1.0/images', methods=['GET'])
def get_images():
    return jsonify({'images': images})

@app.route('/faces/api/v1.0/images/<int:image_id>/faces', methods=['GET'])
def get_image(image_id):
    image = next( (image for image in images if image['id'] == image_id), None )
    if image is None:
        abort(404)
    return jsonify({'image': image})

def extract_image(what):
    if not request.json: # or not 'title' in request.json:
        log.error('no JSON data')
        abort(400)
    data = json.loads(request.json)
    image = data[what]
    if image is None:
        log.error('JSON does not have "image" data')
        abort(400)
    log.error('decoding image data')
    jpg_original = base64.b64decode(image)
    log.error('creating image data')
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8);
    log.error('decoding image')
    image = cv2.imdecode(jpg_as_np, flags=1)
    log.error('translating to gray image')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     log.error('saving image to a file')
#     cv2.imwrite('output.png', image)
    return image

@app.route('/faces/api/v1.0/images', methods=['POST'])
def post_image():
    log.info('POST')
    id = 0
    if len(images) > 0:
        id = images[-1]['id'] + 1
    log.info('new image id: ' + str(id))

    image = extract_image(u'image')    
    face_records = []
    log.error('getting face agent')
    agent = get_face_agent()
    log.error('recognizing faces')
    faces = agent.recognize(image)
    log.error('compiling faces data')
    for face in faces:
        record = {
            'location' : face['location'],
            'size' : face['size'],
            'name' : agent.decode_face_id(face['predicted']),
            'confidence' : face['confidence']
        }
        face_records.append(record)
    record = {
        'id': id,
        'faces': face_records
    }
    images.append(record)
    log.error('done')
    return jsonify(record), 201

@app.route('/faces/api/v1.0/face', methods=['POST'])
def post_face():
    log.info('POST')
    image = extract_image(u'face')    
    log.error('getting face agent')
    agent = face_agent #get_face_agent()
    log.error('recognizing the face')
    record = agent.predict(image)
    log.error('recognized '+str(record))
    log.error('done')
    return jsonify(record), 201

@app.route('/faces/api/v1.0/images/<int:image_id>', methods=['DELETE'])
def delete_task(image_id):
    to_remove = [i for i, image in enumerate(images) if image['id']==image_id]
    if len(to_remove) == 0:
        abort(404)
    for index in reversed(to_remove): # start at the end to avoid recomputing offsets
        del images[index]
    return jsonify({'result': True})

if __name__ == '__main__':
    app.run(debug=True)
