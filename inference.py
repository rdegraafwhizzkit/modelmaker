import tflite_runtime.interpreter as tflite

model_path = './models/'
interpreter = tflite.Interpreter(
    model_path,
    experimental_delegates=[tflite.load_delegate('libedgetpu.1.dylib')]
)
