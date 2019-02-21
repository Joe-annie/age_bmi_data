import pymongo
import numpy as np
import cv2
from predictor import predict_age


def connect(col_name='image', db_name='face_new', uri='mongodb://192.168.3.200:27019'):
    client = pymongo.MongoClient(uri)
    db = client.get_database(db_name)
    return db.get_collection(col_name)


def pre_process_from_buf(buf, face_size=64):
    try:
        img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), -1)
        face = cv2.resize(img, (face_size, face_size))
        face = np.expand_dims(face, axis=0)
        return {
            'code': 0,
            'face': face
        }
    except:
        return {
            'code': -1
        }


def batch_update():
    col = connect('metric')
    col_image = connect('image')
    for c in col.find({}, {'md5': 1}):
        md5 = c.get('md5')
        a_dict = col_image.find_one({'md5': md5}, {'face_data': 1})
        if a_dict:
            face_data = a_dict.get('face_data')
            if face_data:
                r = pre_process_from_buf(face_data)
                if r['code'] == 0:
                    face = r['face']
                    age_result = predict_age(face)
                    print(age_result)
                    if age_result['code'] == 0:
                        age_predict = age_result['age']
                        col.update_one({'md5': md5}, {'$set': {'age_predict': age_predict}})
                        print('update {} and set age_predict to {:d}'.format(md5, age_predict))


def main():
    batch_update()


if __name__ == '__main__':
    main()
