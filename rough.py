import tensorflow
from tensorflow.keras.models import load_model # type: ignore

model = load_model("models/disaster.h5")
print(model.input_shape)