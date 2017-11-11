
import elk
import types
import cv2
import os
import numpy as np
import math
from PIL import Image
import logging as log
from app.homage.camera import Camera.Camera
from app.homage.face.FaceAligner import FaceAligner  

log.basicConfig(level=log.INFO)#filename='webcam.log',level=log.INFO)

class FaceImage(elk.Elk):
    image = elk.ElkAttribute(mode='rw', type=(types.NoneType,np.ndarray),
        builder='_image', lazy=True )
    filename = elk.ElkAttribute(mode='ro', type=str)

    def _image(self):
        if self.filename:
            image = self._load_from_file(self.filename)
            return image
        return Image()
    
    def _load_from_file(self,filename):
        log.info("reading: "+filename)
        image = Image.open(filename)
        image = image.convert('L')
        image = np.array(image, 'uint8')
        return image

    def gray(self):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return image

    def resize(self,size):
        h,w = self.image.shape[:2]
        sh,sw = size
        if h > sh or w > sw: # shrinking
            interp = cv2.INTER_AREA
        else:
            interp = cv2.INTER_CUBIC
        aspect = w/h
        if aspect > 1: # horizontal image
            new_w = sw
            new_h = np.round(new_w/aspect).astype(int)
            pad_vert = (sh-new_h)/2
            pad_top, pad_bot = np.floor(pad_vert).astype(int),np.ceil(pad_vert).astype(int)
            pad_left, pad_right = 0, 0
        elif aspect < 1: # vertical image
            new_h = sh
            new_w = np.round(new_h*aspect).astype(int)
            pad_horz = (sw-new_w)/2
            pad_left, pad_right = np.floor(pad_horz).astype(int),np.ceil(pad_horz).astype(int)
            pad_top, pad_bot = 0, 0
        else:
            new_h, new_w = sh, sw
            pad_left, pad_right, pad_top, pad_bot = 0,0,0,0
        image = cv2.resize(self.image, (new_w, new_h), interpolation=interp)
        image = cv2.copyMakeBorder( image, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=0)
        self.image = image
        return image

    def align_eyes(self,left,right,image=None):
        if image is None:
            image = self.image

        (ex,ey,ew,eh) = left        
        eye_left=(ex,ey)
        (ex,ey,ew,eh) = right
        eye_right=(ex,ey)
        scale=(0.15,0.10)
        size = (image.shape[0], image.shape[1])
        dest_sz = (image.shape[0], image.shape[1])
        #eye_left=(0,0), eye_right=(0,0), offset_pct=(0.25,0.25), dest_sz = (250,250)):

        # rotate
        eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
        rotation = math.atan2(float(eye_direction[1]), float(eye_direction[0]))
        rotation = rotation * 180 / math.pi
        dist = math.hypot(eye_left[0] - eye_right[0], eye_left[1] - eye_right[1])
    
        sz = image.shape
        if len(sz) > 2: sz = sz[:2]
   
        mat = cv2.getRotationMatrix2D(eye_left, rotation, 1.0)
        result = cv2.warpAffine(image, mat, sz, flags = cv2.INTER_CUBIC)

        # cut    
#         dest_h = math.floor(float(dest_scale[0])*dest_sz[0])
#         dest_v = math.floor(float(dest_scale[1])*dest_sz[1])
# #        reference = dest_sz[0] - 2.0*offset_h
#         reference = dest_sz[0] - 2.0*offset_h
#         scale = float(dist) / float(reference)
#         print( dist, reference )
#         print( scale )

        crop_xy = (size[0]*scale[0], size[1]*scale[1])
        #crop_size = (dest_sz[0]*scale, dest_sz[1]*scale)
        crop_size = ( size[0]*( 1 - 2* scale[0] ), size[1]*( 1 - 2* scale[1] ))
        result = result[
            int(crop_xy[1]):int(crop_xy[1]+crop_size[1]),
            int(crop_xy[0]):int(crop_xy[0]+crop_size[0])
        ]
    
        self.image = result
        return result

    def normalize(self):
        image = self.resize(self.image, (100,100))
        image = cv2.equalizeHist(image)
        image = cv2.bilateralFilter(image, 5, 75, 75)
        self.image = image
        return image

if __name__ == "__main__":
    f = FaceImage( filename='')
    (result,image) = f.crop()
    cv2.imshow('original', result)
    cv2.imshow('new', image)
    k = cv2.waitKey(3000) & 0xff
    print("aaaaaa\n")
#     if k == 27:
#         break
