import onnx
from onnx2keras import onnx_to_keras
import torch
import numpy as np

x = torch.randn(1, 3, 480, 640, requires_grad=False)

# Load ONNX model
onnx_model = onnx.load('yolov5n-0.5.onnx')
input_all = [node.name for node in onnx_model.graph.input]
# Call the converter (input will be equal to the input_names parameter that you defined during exporting)

onnx.helper.printable_graph(onnx_model.graph)
import onnxruntime as ort

ort_session = ort.InferenceSession('yolov5n-0.5.onnx')
input_shape = (1, 3, 640, 640)
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

outputs = ort_session.run(
    None,
    {'input': input_data}
)
print(outputs)

# from onnx_tf.backend import prepare
#
# tf_rep = prepare(onnx_model)
# tf_rep.export_graph("yolo")
import tensorflow as tf

model = tf.saved_model.load("yolo")
model.trainable = False

# input_tensor = tf.random.uniform([1, 3, 640, 640])
out = model(**{'input': input_data})

print(out)

# converter = tf.lite.TFLiteConverter.from_saved_model("yolo")
# tflite_model = converter.convert()
# with open("converted_model.tflite", "wb") as f:
#     f.write(tflite_model)

# # Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data
# input_shape = input_details[0]['shape']
# input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# get_tensor() returns a copy of the tensor data
# use tensor() in order to get a pointer to the tensor
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
