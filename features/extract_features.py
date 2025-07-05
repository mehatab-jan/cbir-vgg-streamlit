import os
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tqdm import tqdm

# Load VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

# Dataset and output path
dataset_path = "dataset/"
output_dir = "features/"
os.makedirs(output_dir, exist_ok=True)

features = []
filenames = []

# Loop through images
for root, dirs, files in os.walk(dataset_path):
    for file in tqdm(files):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(root, file)
            img = image.load_img(path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            feature = model.predict(img_array)[0]
            features.append(feature)
            filenames.append(path)

# Save the features
np.save(os.path.join(output_dir, "deep_features.npy"), features)
np.save(os.path.join(output_dir, "filenames.npy"), filenames)
