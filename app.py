from flask import Flask, request
from flask_redis import FlaskRedis
from flask_api import status
import models.facial_recognition as fr
import json
import os
from io import StringIO
import numpy as np

REDIS_URL = "redis://:password@localhost:6379/0"
app = Flask(__name__)
redis_client = FlaskRedis(app)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/registerFace', methods=['POST'])
def registerFace():
    imgPath = None
    try:
        res = request.form
        image = request.files['image']
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], str(res['name'])) + '.jpg')

        # type of ndarray
        imgPath = './' + app.config['UPLOAD_FOLDER'] + '/' + str(res['name']) + '.jpg'
        embeddings = fr.get_embeddings([imgPath])
        faceData = json.dumps(','.join(map(str, embeddings[0])))
        redis_client.set(res['name'], faceData)
        return {
                   "ret": 0,
                   "msg": "Facial image has been successfully registered"
               }, status.HTTP_200_OK
    except Exception as e:
        return {
                   "ret": 400,
                   "msg": str(e)
               }, status.HTTP_400_BAD_REQUEST
    finally:
        if imgPath:
            os.remove(imgPath)



@app.route('/compareFace', methods=['POST'])
def compareFace():
    try:
        keys = redis_client.keys()
        allFaceData = {}

        for i in range(0, len(keys)):
            name = keys[i].decode('utf-8')
            s = StringIO(redis_client.get(name).decode())
            allFaceData[name] = np.genfromtxt(s, delimiter=",")
            allFaceData[name] = np.nan_to_num(allFaceData[name])

        res = request.form
        targetName = res['name']

        try:
            targetFacialData = allFaceData[targetName]
        except KeyError as e:
            return {
                       "ret": 0,
                       "msg": 'The name provided is not in our record'
                   }, status.HTTP_404_NOT_FOUND

        if targetFacialData.any():
            image = request.files['image']
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], str(res['name'])) + '_temp.jpg')
            embeddings = fr.get_embeddings(['./' + app.config['UPLOAD_FOLDER'] + '/' + str(res['name']) + '_temp.jpg'])

            ## Boolean - check if the submitted image matches the previously registered face
            faceMatch = fr.find_similar_person(embeddings, targetFacialData, targetName, redis_client)
            os.remove('./' + app.config['UPLOAD_FOLDER'] + '/' + str(res['name']) + '_temp.jpg')

            if faceMatch:
                return {
                           "ret": 0,
                           "msg": "Facial recognition success!"
                       }, status.HTTP_200_OK
            else:
                return {
                           "ret": 0,
                           "msg": "Facial recognition failed!"
                       }, status.HTTP_403_FORBIDDEN
        else:
            return {
                       "ret": 0,
                       "msg": 'The name provided is not in our record'
                   }, status.HTTP_404_NOT_FOUND
    except Exception as e:
        return {
                   "ret": 400,
                   "msg": str(e)
               }, status.HTTP_400_BAD_REQUEST


if __name__ == '__main__':
    app.run()
