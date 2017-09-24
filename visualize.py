import cv2
from keras.models import model_from_json
from keras import backend as K
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import numpy as np

def load_model():
    #load the model
    try:
        model_file = open('model4.json', 'r')
        loaded_model = model_file.read()
        model_file.close()
        model = model_from_json(loaded_model)
        #load weights into the model
        model.load_weights("weights4.hdf5")
        print("Model loaded successfully")
        return model
    except:
        print("Error while loading model. Terminating")
        return None

model = load_model()


def visualize( img, layer_index=0, filter_index=0 ,all_filters=False ):
    
    act_fun = K.function([model.layers[0].input, K.learning_phase()], 
                                  [model.layers[layer_index].output,])
    x=img_to_array(img)
    img = cv2.cvtColor( x, cv2.COLOR_RGB2GRAY )
    img=img.reshape(img.shape+(1,))
    img=img.reshape((1,)+img.shape)
    img = act_fun([img,0])[0]
    
    if all_filters:
        fig=plt.figure(figsize=(7,7))
        filters = len(img[0,0,0,:])
        for i in range(filters):
                plot = fig.add_subplot(6, 6, i+1)
                plot.imshow(img[0,:,:,i],'gray')
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
        plt.tight_layout()
    else:
        img = np.rollaxis(img, 3, 1)
        img=img[0][filter_index]
        print(img.shape)
        cv2.imshow("",img)
        
        
visualize(img = load_img('15.jpg',target_size=(33,50)), layer_index =3 ,all_filters = True)
