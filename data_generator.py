from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_train_valid_generator(train_dir, valid_dir, rotation_range, height_shift_range,
                                 width_shift_range, zoom_range, horizontal_flip, train_batch_size,
                                 valid_batch_size):

    train_datagen = ImageDataGenerator(rescale=1. / 255.,
                                       rotation_range=rotation_range,
                                       height_shift_range=height_shift_range,
                                       width_shift_range=width_shift_range,
                                       zoom_range=zoom_range,
                                       horizontal_flip=horizontal_flip) 
 
    valid_datagen = ImageDataGenerator(rescale=1.0 / 255.)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        batch_size=train_batch_size,
                                                        class_mode='categorical',
                                                        target_size=(224, 224))


    validation_generator = valid_datagen.flow_from_directory(valid_dir,
                                                             batch_size=valid_batch_size,
                                                             class_mode='categorical',
                                                             target_size=(224, 224))
    return train_generator, validation_generator