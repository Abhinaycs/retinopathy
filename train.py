import json
import datetime
import os

from model import densenet121_tf
from data_generator import create_train_valid_generator
from tensorflow_addons.metrics import CohenKappa

# Load config.json instead of using command-line arguments
def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)

def train_model(train_dir, valid_dir, rotation_range, height_shift_range,
                width_shift_range, zoom_range, horizontal_flip, train_batch_size,
                valid_batch_size, epochs, save_model=True):

    # Check if directories exist
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.exists(valid_dir):
        raise FileNotFoundError(f"Validation directory not found: {valid_dir}")

    train_generator, validation_generator = create_train_valid_generator(
        train_dir, valid_dir, rotation_range, height_shift_range,
        width_shift_range, zoom_range, horizontal_flip, train_batch_size,
        valid_batch_size
    )

    print(f"Training samples found: {train_generator.samples}")
    print(f"Validation samples found: {validation_generator.samples}")

    if train_generator.samples == 0 or validation_generator.samples == 0:
        raise ValueError("No images found in training or validation directories!")

    model = densenet121_tf()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                  metrics=[CohenKappa(num_classes=5, sparse_labels=False, weightage='quadratic')])

    history = model.fit(train_generator, epochs=epochs,
                        validation_data=validation_generator, verbose=1)

    if save_model:
        today_date = str(datetime.date.today())
        os.makedirs("assets", exist_ok=True)  # Ensure 'assets' directory exists
        model.save(f'assets/densenet121_{today_date}')

if __name__ == '__main__':
    config = {
        "train_dir": "B:/organized_train_images/train_images",
        "valid_dir": "B:/organized_train_images/valid_images",
        "rotation_range": 35,
        "height_shift_range": 0.2,
        "width_shift_range": 0.15,
        "zoom_range": 0.2,
        "horizontal_flip": True,
        "train_batch_size": 16,
        "valid_batch_size": 16,
        "epochs": 20
    }

    train_model(
        train_dir=config["train_dir"],
        valid_dir=config["valid_dir"],
        rotation_range=config["rotation_range"],
        height_shift_range=config["height_shift_range"],
        width_shift_range=config["width_shift_range"],
        zoom_range=config["zoom_range"],
        horizontal_flip=config["horizontal_flip"],
        train_batch_size=config["train_batch_size"],
        valid_batch_size=config["valid_batch_size"],
        epochs=config["epochs"]
    )
