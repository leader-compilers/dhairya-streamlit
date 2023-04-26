from matplotlib.animation import FuncAnimation
import cv2
import tensorflow as tf
import streamlit.components.v1 as components
import random
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from keras import backend
from keras.applications.vgg16 import VGG16
tf.compat.v1.disable_eager_execution()

class_names = ['mountain', 'street', 'glacier', 'buildings', 'sea', 'forest']
image_paths = {'mountain': ['images/m1.jpg', 'images/m2.jpg', 'images/m3.jpg', 'images/m4.jpg', 'images/m5.jpg'], 'street': ['images/st1.jpg', 'images/st2.jpg', 'images/st3.jpg', 'images/st4.jpg', 'images/st5.jpg'], 'glacier': ['images/g1.jpg', 'images/g2.jpg', 'images/g3.jpg', 'images/g4.jpg',
                                                                                                                                                                                                                                    'images/g5.jpg'], 'buildings': ['images/b1.jpg', 'images/b2.jpg', 'images/b3.jpg', 'images/b4.jpg', 'images/b5.jpg'], 'sea': ['images/s1.jpg', 'images/s2.jpg', 'images/s3.jpg', 'images/s4.jpg', 'images/s5.jpg'], 'forest': ['images/f1.jpg', 'images/f2.jpg', 'images/f3.jpg', 'images/f4.jpg', 'images/f5.jpg']}


# Create a streamlit app
st.title("Saliency Maps for Image Classification")
st.write(
    "We have developed this app to help you analyse how the saliency map activations of the image changes on passing through each layer of any CNN architechture. We used the pre-trained ResNet model and the kaggle image dataset with 6 classes, namely mountain, street, glacier, buildings, sea, and forest."
)

st.write(
    "You can use the dropdown to select any class, and the animation of the saliency maps correspoding to the test image of that class will be displayed. PS: The animation may take a small amount of time to load."
)


with st.sidebar:
    class_name = st.selectbox('Select Class', class_names)
    file_list = image_paths[class_name]
    image_path = random.choice(file_list)

# st.write(image_path)
image = cv2.imread(image_path)
fig, ax = plt.subplots()
plt.title(f'Random Test Image of {class_name}')
plt.axis('off')
plt.imshow(image)
st.pyplot(fig)

# Load a pre-trained model
model = VGG16(weights='imagenet', include_top=False)

input_shape = (224, 224)
x = cv2.resize(image, input_shape)
# only taking 25 layers out of 175
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
maps_lst = []

for layer_name in layer_dict:
    # Get the output of the current layer and the input to the model
    layer_output = layer_dict[layer_name].output
    layer_input = model.input
    grads = backend.gradients(layer_output, layer_input)[0]
    # grads /= (backend.sqrt(backend.mean(backend.square(grads))) +
    #           backend.epsilon())
    saliency_fn = backend.function([layer_input], [grads])
    maps_lst.append(saliency_fn([x[np.newaxis, ...]])[0][0])


fig1, ax1 = plt.subplots()
# Create a gif of the saliency maps


def update(frame):
    ax1.clear()
    ax1.imshow(maps_lst[frame], cmap='viridis')
    ax1.set_title(f"Layer {frame}")


anim = FuncAnimation(fig1, update, frames=25, interval=10000)


with open("animation.html", "w") as f:
    print(anim.to_html5_video(), file=f)

HtmlFile = open("animation.html", "r")
source_code = HtmlFile.read()
components.html(source_code, height=500, width=900)
