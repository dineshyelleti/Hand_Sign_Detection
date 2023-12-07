import inspect
from keras.models import load_model 

model = load_model("Model\keras_model.h5",)

source_code = inspect.getsource(model.predict)

with open('output.txt', 'w') as file:

    file.write(source_code)



