import tflite_runtime.interpreter as tflite
from pycoral.adapters import common, classify
from pathlib import Path
from PIL import Image
from os.path import expanduser

script_dir = Path(__file__).parent.absolute()
model_path = f'{script_dir}/models/model_edgetpu.tflite'
image_file = f'{expanduser("~")}/.keras/datasets/flower_photos/tulips/15275504998_ca9eb82998.jpg'

interpreter = tflite.Interpreter(
    model_path,
    experimental_delegates=[tflite.load_delegate('libedgetpu.1.dylib')]
)
interpreter.allocate_tensors()
size = common.input_size(interpreter)

image = Image.open(image_file).convert('RGB').resize(size, Image.Resampling.LANCZOS)
common.set_input(interpreter, image)
interpreter.invoke()
classes = classify.get_classes(interpreter, top_k=1)
print(classes)
