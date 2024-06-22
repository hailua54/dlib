import numpy as np
import dlib
def test():
  print(dlib.__version__)
  print('dlib.DLIB_USE_CUDA v1 ', dlib.DLIB_USE_CUDA)
  img = dlib.load_rgb_image("../../../examples/faces/Tom_Cruise_avp_2014_4.jpg")
  
  # Load the face detection model
  detector = dlib.get_frontal_face_detector()

  # Detect faces in the image
  faces = detector(img)
  
  print("faces len", len(faces))
  
test()