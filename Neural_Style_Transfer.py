#%%
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.python.keras.applications.vgg19 import VGG19,preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img,img_to_array
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
%matplotlib inline
# %%
model = VGG19(include_top=False,weights='imagenet')
model.trainable = False
model.summary()
# %%
def load_and_process_image(image_path):
    img = load_img(image_path)
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img,axis=0)
    return img
# %%
def deprocess(x):
    x[:,:,0] += 103.939
    x[:,:,1] += 116.779
    x[:,:,2] += 123.68
    x = x[:,:,::-1]
    x = np.clip(x,0,255).astype('uint8')
    return x

def display_image(image):
    if len(image.shape) == 4:
        img = np.squeeze(image,axis=0)
    img  = deprocess(img)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    return 
# %%
img = load_and_process_image('style1.jpg')
display_image(img)
img = load_and_process_image('content1.jpg')
display_image(img)
# %%
style_layers = ['block1_conv1','block3_conv1','block5_conv1']
content_layers = 'block5_conv2'

content_model = Model(inputs=model.input,outputs=model.get_layer(content_layers).output)
style_models = [Model(inputs=model.input,outputs=model.get_layer(layer).output)for layer in style_layers]
# %%
def content_cost(content,generated):
    a_C = content_model(content)
    a_G = content_model(generated)
    cost = tf.reduce_mean(tf.square(a_C-a_G))
    return cost
# %%
def gram_matrix(A):
    channels = int(A.shape[-1])
    a = tf.reshape(A,[-1,channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a,a,transpose_a=True)
    return gram/tf.cast(n,tf.float32)
# %%
lam = 1.0/len(style_models)
def style_cost(style,generated):
    J_style = 0
    for style_model in style_models:
        a_S = style_model(style)
        a_G = style_model(generated)
        GS = gram_matrix(a_S)
        GG = gram_matrix(a_G)
        current_cost = tf.reduce_mean(tf.square(GS-GG))
        J_style += current_cost * lam
    return J_style
# %%
generated_images = []
def training_loop(content_path,style_path,iterations=20,a=10.,b=20.):
    content = load_and_process_image(content_path)
    style = load_and_process_image(style_path)
    generated = tf.Variable(content,dtype=tf.float32)
    opt = tf.optimizers.Adam(learning_rate=7.)
    best_cost = 1e12+0.1
    best_image = None
    start_time = time.time()
    for i in range(iterations):
        with tf.GradientTape() as tape:
            J_content = content_cost(content,generated)
            J_style = style_cost(style,generated)
            J_total = a*J_content + b*J_style
        grads = tape.gradient(J_total,generated)
        opt.apply_gradients([(grads,generated)])

        if J_total < best_cost:
            best_cost = J_total
            best_image = generated.numpy()
        
        if i % int(iterations/10) == 0:
            time_taken = time.time() - start_time
            print(i,J_total,time_taken)
    return best_image
# %%
final = training_loop('content1.jpg','style1.jpg')
display_image(final)
# %%
