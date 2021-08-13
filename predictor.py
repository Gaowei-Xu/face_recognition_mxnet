from __future__ import print_function
import os
import time
import mxnet as mx
import cv2
import base64
from face import Face
import numpy as np
import insightface
from insightface.utils import face_align


# MODEL_ROOT_DIR = '/opt/ml/model'
MODEL_ROOT_DIR = './models'

# options = ["retinaface_r50_v1", "retinaface_mnet025_v2"]
FACE_DETECTOR_MODEL = os.environ.get("FACE_DETECTOR_MODEL", "retinaface_mnet025_v2")

# options = ["MobileFaceNet", "LResNet34E-IR", "LResNet50E-IR", "LResNet100E-IR"]
FACE_REPRESENT_MODEL = os.environ.get("FACE_REPRESENT_MODEL", "MobileFaceNet")


# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.
class FaceRecognizerService(object):
    # class attributes
    face_detector = None
    face_embedding_model = None

    ctx = mx.cpu() if mx.context.num_gpus() == 0 else mx.gpu()

    # face representation configuration
    face_size = (112, 112)

    if FACE_REPRESENT_MODEL == 'LResNet100E-IR':
        FACE_REPRESENT_MODEL_PREFIX = os.path.join(MODEL_ROOT_DIR, 'model-r100-ii/model')
    elif FACE_REPRESENT_MODEL == 'LResNet50E-IR':
        FACE_REPRESENT_MODEL_PREFIX = os.path.join(MODEL_ROOT_DIR, 'model-r50-am-lfw/model')
    elif FACE_REPRESENT_MODEL == 'LResNet34E-IR':
        FACE_REPRESENT_MODEL_PREFIX = os.path.join(MODEL_ROOT_DIR, 'model-r34-amf/model')
    elif FACE_REPRESENT_MODEL == 'MobileFaceNet':
        FACE_REPRESENT_MODEL_PREFIX = os.path.join(MODEL_ROOT_DIR, 'model-y1-test2/model')
    else:
        FACE_REPRESENT_MODEL_PREFIX = 'None'

    @classmethod
    def load_model(cls):
        """
        Get the face detector and face representation model for this instance, loading it if it's not already loaded.
        :return:
        """
        # face detector model
        if cls.face_detector is None:
            cls.face_detector = insightface.model_zoo.get_model(
                name=FACE_DETECTOR_MODEL,
                root=MODEL_ROOT_DIR
            )
            ctx_id = -1 if mx.context.num_gpus() == 0 else 0
            cls.face_detector.prepare(ctx_id=ctx_id)

        # face representation (embedding vector representation) model
        if cls.face_embedding_model is None:
            sym, arg_params, aux_params = mx.model.load_checkpoint(
                prefix=cls.FACE_REPRESENT_MODEL_PREFIX, epoch=0)
            all_layers = sym.get_internals()
            sym = all_layers['fc1_output']
            cls.face_embedding_model = mx.mod.Module(symbol=sym, context=cls.ctx, label_names=None)
            cls.face_embedding_model.bind(data_shapes=[('data', (1, 3, cls.face_size[0], cls.face_size[1]))])
            cls.face_embedding_model.set_params(arg_params, aux_params)

        return cls.face_detector, cls.face_embedding_model

    @classmethod
    def detect_and_represent(cls, raw_input_image, threshold=0.70):
        face_detector, _ = cls.load_model()

        height, width, _ = raw_input_image.shape
        short_size = height if height < width else width
        scale = 1.0 if short_size < 480.0 else 480.0 / short_size

        # detection inference
        bbox_list, pts5_list = face_detector.detect(raw_input_image, threshold=threshold, scale=scale)

        face_list = list()

        if bbox_list.shape[0] == 0:
            return face_list

        for face_idx in range(len(bbox_list)):
            bbox = bbox_list[face_idx]
            pts5 = pts5_list[face_idx]

            aligned_target_face = face_align.norm_crop(raw_input_image, pts5)
            embedding_vector = cls.get_feature(aligned_target_face)

            face = Face(
                bbox=[float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                aligned_face_data=aligned_target_face,
                confidence=float(bbox[-1]),
                landmarks={
                    'eyeLeft': [float(pts5[0][0]), float(pts5[0][1])],
                    'eyeRight': [float(pts5[1][0]), float(pts5[1][1])],
                    'nose': [float(pts5[2][0]), float(pts5[2][1])],
                    'mouthLeft': [float(pts5[3][0]), float(pts5[3][1])],
                    'mouthRight': [float(pts5[4][0]), float(pts5[4][1])],
                },
                representation=embedding_vector
            )

            face_list.append(face)
        return face_list

    @classmethod
    def get_feature(cls, aligned_face):
        _, face_embedding_model = cls.load_model()

        a = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        a = np.transpose(a, (2, 0, 1))
        input_blob = np.expand_dims(a, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data, ))
        face_embedding_model.forward(db, is_train=False)
        emb = face_embedding_model.get_outputs()[0].asnumpy()[0]
        norm = np.sqrt(np.sum(emb * emb) + 0.00001)
        emb /= norm
        return emb


def invoke_face_recognition_service(image_base64_enc):
    # image decode
    t1 = time.time()
    image_data = cv2.imdecode(np.frombuffer(base64.b64decode(image_base64_enc), np.uint8), cv2.IMREAD_COLOR)
    height, width, channels = image_data.shape

    # face detection, alignment and represent
    t2 = time.time()
    faces = FaceRecognizerService.detect_and_represent(image_data)

    # response construct
    t3 = time.time()
    body = {
        "image_height": height,
        "image_width": width,
        "image_channels": channels,
        "faces": list()
    }

    for index, face in enumerate(faces):
        [x_min, y_min, x_max, y_max] = face.bbox
        confidence = face.confidence
        landmarks = face.landmarks
        representation = face.representation

        body["faces"].append(
            {
                "bbox": [x_min, y_min, x_max, y_max],
                "confidence": confidence,
                "landmarks": landmarks,
                "representation": representation
            }
        )
    t4 = time.time()

    print('Total Time Cost = {} ms'.format(1000.0 * (t4 - t1)))
    print('    Decode Time Cost = {} ms'.format(1000.0 * (t2 - t1)))
    print('    Face Detection & Alignment & Represent (Face Amount = {}) Time Cost = {} ms'.format(len(faces), 1000.0 * (t3 - t2)))
    print('    Construct Response Body = {} ms'.format(1000.0 * (t4 - t3)))

    return body["faces"]


def get_base64_encoding(test_image_full_path):
    with open(test_image_full_path, "rb") as f:
        data = f.read()
        image_base64_enc = base64.b64encode(data)
        image_base64_enc = str(image_base64_enc, 'utf-8')

    return image_base64_enc


if __name__ == '__main__':
    processor = FaceRecognizerService()
    processor.load_model()

    src_image = get_base64_encoding(test_image_full_path='./test_imgs/test_1_source.jpeg')
    tgt_image = get_base64_encoding(test_image_full_path='./test_imgs/test_1_target.jpeg')
    no_face_image = get_base64_encoding(test_image_full_path='./test_imgs/noface.jpeg')

    source_faces = invoke_face_recognition_service(image_base64_enc=src_image)
    target_faces = invoke_face_recognition_service(image_base64_enc=tgt_image)
    dummy_faces = invoke_face_recognition_service(image_base64_enc=no_face_image)

    # import matplotlib.pyplot as plt
    # import numpy as np
    # from distance import *
    #
    # src_image_data = cv2.imdecode(np.frombuffer(base64.b64decode(src_image), np.uint8), cv2.IMREAD_COLOR)
    # tgt_image_data = cv2.imdecode(np.frombuffer(base64.b64decode(tgt_image), np.uint8), cv2.IMREAD_COLOR)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 2, 1)
    # plt.imshow(src_image_data[:, :, ::-1])
    # plt.axis('off')
    #
    # ax = fig.add_subplot(1, 2, 2)
    # plt.imshow(tgt_image_data[:, :, ::-1])
    # colors = ['blue', 'yellow', 'red', 'green', 'black', 'cyan', 'pink']
    #
    # for src_face in source_faces:
    #     for index, tgt_face in enumerate(target_faces):
    #         src_face_vector = np.array(src_face["representation"])
    #         tgt_face_vector = np.array(tgt_face["representation"])
    #
    #         distance_cosine = find_cosine_distance(src_face_vector, tgt_face_vector)
    #         distance_euclidean_l2 = find_euclidean_distance(l2_normalize(src_face_vector), l2_normalize(tgt_face_vector))
    #
    #         print('Cosine Distance = {} (Threshold = {})'.format(distance_cosine, find_threshold('cosine')))
    #         print('Euclidean L2 Distance = {} (Threshold = {})\n'.format(distance_euclidean_l2, find_threshold('euclidean_l2')))
    #
    #         [x_min, y_min, x_max, y_max] = tgt_face["bbox"]
    #         rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_max, fill=False, edgecolor=colors[index])
    #         ax.add_patch(rect)
    #
    # plt.axis('off')
    # plt.show()
