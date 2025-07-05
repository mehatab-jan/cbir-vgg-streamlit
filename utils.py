import numpy as np
import os
import cv2
import streamlit as st

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_top_k_similar(query_feat, dataset_feats, image_paths, k=5):
    sims = [cosine_similarity(query_feat, feat) for feat in dataset_feats]
    top_indices = np.argsort(sims)[::-1][:k]
    return [image_paths[i] for i in top_indices]

def load_image_paths(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('jpg', 'png', 'jpeg'))]

def show_images(image_paths):
    for path in image_paths:
        st.image(path, width=200, caption=os.path.basename(path))
