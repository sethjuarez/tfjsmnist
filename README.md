# TensorFlow.js MNIST Example
Sample for testing tensorflow.js. Started with this awesome [codepen](https://codepen.io/getflourish/pen/EyqxYE) for drawing on a canvas and made some modifications.

# Running TensorFlow Training
Train the digit recognizer model:
```
python train.py -e 12
```

# Converting Model
Install utility to convert model trained model to tensorflow.js:
```
pip install tensorflowjs
```
Then convert the newly created model (make sure to move the `digits.pb` file to the right spot):
```
tensorflowjs_converter digits.pb model --input_format tf_frozen_model --output_format tensorflowjs --output_node_names Model/prediction
```