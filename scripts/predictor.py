import os
import numpy as np
import cv2
# import tensorflow as tf

from mtcnn.mtcnn import MTCNN

from SSRNET_model import SSR_net_general, SSR_net

AGE_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ssrnet_3_3_3_64_1.0_1.0.h5')
GENDER_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wiki-gender-ssrnet_3_3_3_64_1.0_1.0.h5')


def make_bgr(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        return img


def to_buffer(img, ext='.jpg'):
    _, img_encode = cv2.imencode(ext, img)
    return np.array(img_encode).tostring()


def to_cv_mat(f):
    if hasattr(f, 'read'):
        buf = f.read()
    else:
        buf = f
    img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), -1)
    return make_bgr(img)


def create_gender_model(model_path=GENDER_MODEL_PATH):
    filename = os.path.basename(model_path)
    filename_no_ext = os.path.splitext(filename)[0]
    model_name, stage_num0, stage_num1, stage_num2, img_size, lambda_local, lambda_d = filename_no_ext.split('_')
    stage_num = [int(e) for e in (stage_num0, stage_num1, stage_num2)]
    img_size = int(img_size)
    lambda_local = float(lambda_local)
    lambda_d = float(lambda_d)
    model = SSR_net_general(img_size, stage_num, lambda_local, lambda_d)()
    model.load_weights(model_path)
    model.save('wiki-gender-ssrnet_3_3_3_64_1.0_1.0.h5')
    return model


def create_age_model(model_path=AGE_MODEL_PATH):
    filename = os.path.basename(model_path)
    filename_no_ext = os.path.splitext(filename)[0]
    model_name, stage_num0, stage_num1, stage_num2, img_size, lambda_local, lambda_d = filename_no_ext.split('_')
    stage_num = [int(e) for e in (stage_num0, stage_num1, stage_num2)]
    img_size = int(img_size)
    lambda_local = float(lambda_local)
    lambda_d = float(lambda_d)
    model = SSR_net(img_size, stage_num, lambda_local, lambda_d)()
    model.load_weights(model_path)
    model.save('ssrnet_3_3_3_64_1.0_1.0.h5')
    return model


def pre_process(img, detector=MTCNN(), min_confidence=0.95, face_size=64, expand=0.3, net_in=True, compressed=False):
    img_h, img_w, _ = np.shape(img)
    if compressed:
        img = cv2.resize(img, (1024, int(1024 * img_h / img_w)))
    img_h, img_w, _ = np.shape(img)
    detections = detector.detect_faces(img)
    if len(detections) > 0:
        max_confidence = 0
        x1 = y1 = x2 = y2 = 0
        for i, detection in enumerate(detections):
            rect = detection['box']
            confidence = detection['confidence']
            x1_temp, y1_temp, x2_temp, y2_temp = rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]
            if max_confidence < confidence:
                x1, y1, x2, y2 = x1_temp, y1_temp, x2_temp, y2_temp
                max_confidence = confidence
        if max_confidence < min_confidence:
            return {
                'code': -2
            }
        w, h = x2 - x1, y2 - y1
        xw1 = max(int(x1 - expand * w), 0)
        yw1 = max(int(y1 - expand * h), 0)
        xw2 = min(int(x2 + expand * w), img_w - 1)
        yw2 = min(int(y2 + expand * h), img_h - 1)
        face = img[yw1:yw2 + 1, xw1:xw2 + 1, :]
        if net_in:
            face = cv2.resize(face, (face_size, face_size))
            face = np.expand_dims(face, axis=0)
        return {
            'code': 0,
            'face': face
        }
    else:
        return {
            'code': -1
        }


def pre_process_from_buf(buf, detector=MTCNN(), min_confidence=0.95, face_size=64, expand=0.3):
    img = to_cv_mat(buf)
    return pre_process(img, detector, min_confidence, face_size, expand)


def predict_age_all(img, detector=MTCNN(), model=create_age_model(), min_confidence=0.95, face_size=64, expand=0.3):
    result = pre_process(img, detector, min_confidence, face_size, expand)
    if result['code'] == 0:
        face = result['face']
        predict_age = model.predict(face)
        return {
            'code': 0,
            'age': int(predict_age[0, 0])
        }
    else:
        return result


def predict_gender_all(img, detector=MTCNN(), model=create_gender_model(), min_confidence=0.95, face_size=64, expand=0.3):
    result = pre_process(img, detector, min_confidence, face_size, expand)
    if result['code'] == 0:
        face = result['face']
        predicted_gender = model.predict(face)
        return {
            'code': 0,
            'gender': 1 if predicted_gender < 0.5 else 0
        }
    else:
        return result


def predict_age(face, model=create_age_model()):
    try:
        # with graph.as_default():
        age = model.predict(face)
        return {
            'code': 0,
            'age': int(age[0, 0])
        }
    except Exception as err:
        return {
            'msg': err,
            'code': 1
        }


#
# def predict_age_from_buf(face_buf, model=create_age_model()):
#     face = to_cv_mat(face_buf)
#     return predict_age(face, model)


def predict_gender(face, model=create_gender_model()):
    try:
        # with graph.as_default():
        gender = model.predict(face)
        return {
            'code': 0,
            'gender': 1 if gender < 0.5 else 0
        }
    except Exception as err:
        return {
            'msg': err,
            'code': 1
        }


# def predict_gender_from_buf(face_buf, model=create_gender_model()):
#     face = to_cv_mat(face_buf)
#     return predict_gender(face, model)


def predict(img, detector=MTCNN(), age_model=create_age_model(), gender_model=create_gender_model(),
            min_confidence=0.95, face_size=64, expand=0.3):
    try:
        face_result = pre_process(img, detector, min_confidence, face_size, expand)
        if face_result['code'] == 0:
            face = face_result['face']
            # with graph.as_default():
            age = age_model.predict(face)
            age = int(age)
            # with graph.as_default():
            gender = gender_model.predict(face)
            gender = 1 if gender < 0.5 else 0
            return {
                'code': 0,
                'age': age,
                'gender': gender
            }
        else:
            return {
                'code': face_result['code']
            }
    except Exception as err:
        return {
            'msg': err,
            'code': 10
        }


def predict_from_file(fp, detector=MTCNN(), age_model=create_age_model(), gender_model=create_gender_model(),
                      min_confidence=0.95, face_size=64, expand=0.3):
    with open(fp, 'rb') as f:
        img = to_cv_mat(f)
    return predict(img, detector, age_model, gender_model, min_confidence, face_size, expand)


def predict_from_buf(buf, detector=MTCNN(), age_model=create_age_model(), gender_model=create_gender_model(),
                     min_confidence=0.95, face_size=64, expand=0.3):
    img = to_cv_mat(buf)
    return predict(img, detector, age_model, gender_model, min_confidence, face_size, expand)


if __name__ == '__main__':
    image_path = './lyf1.jpg'
    with open(image_path, 'rb') as f:
        buf = f.read()
    r = pre_process_from_buf(buf)
    print(r)
    if r['code'] == 0:
        face = r['face']
        age_result = predict_age(face)
        print(age_result)
        gender_result = predict_gender(face)
        print(gender_result)
