# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# train_dir = "B:/organized_train_images/train_images"
# valid_dir = "B:/organized_train_images/valid_images"

# datagen = ImageDataGenerator(rescale=1.0 / 255)

# train_generator = datagen.flow_from_directory(
#     train_dir,
#     target_size=(224, 224),
#     batch_size=16,
#     class_mode="categorical"
# )

# # valid_generator = datagen.flow_from_directory(
# #     valid_dir,
# #     target_size=(224, 224),
# #     batch_size=16,
# #     class_mode="categorical"
# # )
# # import pandas as pd

# # df = pd.read_csv("B:/train.csv")
# # print(df.columns)
# # import tensorflow as tf
# # print(tf.config.list_physical_devices('GPU'))
# import tensorflow as tf
# import tensorflow_addons as tfa

# # Define custom_objects dictionary
# custom_objects = {"CohenKappa": tfa.metrics.CohenKappa}

# # Load the model with custom objects
# model = tf.keras.models.load_model(
#     "C:/Users/Abhin/Downloads/Diabetic_Retinopathy_Classification-main/Diabetic_Retinopathy_Classification-main/assets/densenet121_2025-04-04",
#     custom_objects=custom_objects
# )

# # Check if the model loads correctly
# model.summary()


import tensorflow as tf
import tensorflow_addons as tfa

# Define custom objects dictionary
custom_objects = {"CohenKappa": tfa.metrics.CohenKappa}

# Load the trained model
model = tf.keras.models.load_model(
    "C:/Users/Abhin/Downloads/Diabetic_Retinopathy_Classification-main/Diabetic_Retinopathy_Classification-main/assets/densenet121_2025-04-04",
    custom_objects=custom_objects
)

# Save it in HDF5 format
model.save("C:/Users/Abhin/Downloads/Diabetic_Retinopathy_Classification-main/diabetic_retinopathy_model.h5")

print("✅ Model saved as diabetic_retinopathy_model.h5")


