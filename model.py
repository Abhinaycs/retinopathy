
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, Model


def densenet121_tf():
    pre_trained_model = DenseNet121(input_shape=(224, 224, 3),
                                    include_top=False,
                                    weights='imagenet')

    for layer in pre_trained_model.layers:
        layer.trainable = False 

    x = layers.Flatten()(pre_trained_model.output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.2)(x) 
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(5, activation='softmax')(x)

    model = Model(pre_trained_model.input, x)
    return model
