# Here, we are going to calculate the similarity between fed image and the images in the dataset. To this end, we will use pre-calculated face embeddings.
# Then, we will calculate the face embeddings of this new image and finally we'll performe the similarity between them and sort them in decreasing trend.
# Then chosse the highist value. If this value be more than a thershold (0.5), we'll show the results, else the result is Unknown!
# face verification with the VGGFace2 model
import json
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import numpy as np


# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
    # load image from file
    pixels = pyplot.imread(filename)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    if results[0]:
        # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        return face_array
    else:
        return []


# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(filenames):
    # create a vggface model
    # Names = ['Angelina Jolie', 'Goh', 'Elena', 'Amin', 'Rahil', 'Leon', 'Ka Choon', 'Walter']
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    # extract faces
    faces = [extract_face(f) for f in filenames]
    # convert into an array of samples
    samples = asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)
    # perform prediction
    yhat = model.predict(samples)
    # yhat is a numpy array with shape (N,2048) where N is size of filenames
    return yhat


def find_similar_person(embeddings, targetFacialData, name, redis, threshold=0.74):
    score = 1 - cosine(np.array(embeddings), np.array(targetFacialData))
    print('Score obtained is', score)
    if score > threshold:
        avgData = (np.array(targetFacialData) + np.array(embeddings)) / 2
        redis.set(name, json.dumps(','.join(map(str, avgData[0]))))
        return True
    else:
        return False
