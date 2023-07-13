import os
import base64
import numpy as np
import cv2
import time
import grpc
import grpc_predict_v2_pb2
import grpc_predict_v2_pb2_grpc
import classes


grpc_host = os.environ.get('GRPC_HOST', 'modelmesh-serving')
grpc_port = int(os.environ.get('GRPC_PORT', '8033'))
model_name = os.environ.get('MODEL_NAME', 'yolo')
channel = grpc.insecure_channel(f"{grpc_host}:{grpc_port}")
stub = grpc_predict_v2_pb2_grpc.GRPCInferenceServiceStub(channel)

def preprocess(content, size=640):
    return
#     img = np.frombuffer(content, dtype=np.uint8)
#     img = cv2.imdecode(img, cv2.IMREAD_COLOR)  # BGR format
#     # format of data is HWC
#     # add image preprocessing if needed by the model
#     img = cv2.resize(img, (224, 224))
#     img = img.astype('float32')
#     #convert to NCHW
#     img = img.transpose(2,0,1)
#     # normalize to adjust to model training dataset
#     img = preprocess2(img)
#
#
#     img = img.reshape(1,3,size,size)
#     # print(path, img.shape, "; data range:",np.amin(img),":",np.amax(img))
#     return img


def request_inference(img, size=640):
    # request content building
    # inputs = []
    # inputs.append(grpc_predict_v2_pb2.ModelInferRequest().InferInputTensor())
    # inputs[0].name = 'input'
    # inputs[0].datatype = "FP32"
    # inputs[0].shape.extend([1, 3, size, size])
    # arr = img.flatten()
    # inputs[0].contents.fp32_contents.extend(arr)
    #
    # # request building
    # request = grpc_predict_v2_pb2.ModelInferRequest()
    # request.model_name = model_name
    # request.inputs.extend(inputs)
    #
    # response = stub.ModelInfer(request)
    # arr = np.frombuffer(response.raw_output_contents[0], dtype=np.float32)
    # max = np.argmax(arr)
    # class_name = classes.yolo_classes[max]

    return classes.coco_classes[0]


def predict(body):
    base64img = body.get('image')
    img_bytes = base64.decodebytes(base64img.encode())
    data = preprocess(img_bytes)
    class_name = request_inference(data)
    return {'prediction': class_name}
