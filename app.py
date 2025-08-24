import os
import json
import io
from typing import List

import streamlit as st
from PIL import Image
import requests
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt

import torch

from src.infer import predict, load_model

st.set_page_config(page_title='Human Action Recognition', page_icon='🤸', layout='wide')

st.title('🤸 Human Action Recognition — Streamlit App')
st.caption('Dataset: visual-layer/human-action-recognition-vl-enriched')

CKPT = 'artifacts/model_best.pt'

@st.cache_resource
def load_ckpt():
    if os.path.exists(CKPT):
        try:
            model, idx_to_label, image_size = load_model(CKPT)
            return True, (model, idx_to_label, image_size)
        except Exception as e:
            st.warning(f'Failed to load model: {e}')
            return False, None
    return False, None

@st.cache_data(show_spinner=False)
def hf_dataset():
    return load_dataset('visual-layer/human-action-recognition-vl-enriched')['train']

def fetch_image(url: str):
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert('RGB')
    except Exception as e:
        return None

tab1, tab2, tab3 = st.tabs(['🔮 Predict', '🗂 Explore', '📊 Stats'])

with tab1:
    st.header('🔮 Predict on your image')
    has_model, model_bundle = load_ckpt()
    if not has_model:
        st.info('Model weights not found. Train a model first so this tab can make predictions. See README for instructions.')
    col_u, col_url = st.columns(2)
    with col_u:
        file = st.file_uploader('Upload an image', type=['jpg','jpeg','png','webp'])
    with col_url:
        img_url = st.text_input('...or paste an image URL')

    img = None
    if file is not None:
        img = Image.open(file).convert('RGB')
    elif img_url:
        st.write('Downloading image...')
        img = fetch_image(img_url)

    if img is not None:
        st.image(img, caption='Input image', use_container_width=True)
        if has_model and st.button('Predict', use_container_width=True):
            # Temporarily save to RAM buffer for predict()
            buf = io.BytesIO()
            img.save(buf, format='JPEG')
            buf.seek(0)
            # Save to a temp file because infer.predict expects a path or URL
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp.write(buf.read())
                tmp_path = tmp.name
            try:
                topk = 5
                outs = predict(tmp_path, CKPT, topk=topk)
                st.subheader('Top predictions')
                for cls, p in outs:
                    st.write(f'- **{cls}** — {p*100:.2f}%')
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

with tab2:
    st.header('🗂 Dataset Explorer')
    ds = hf_dataset()
    labels = sorted(list(set(ds['image_label'])))
    sel_label = st.selectbox('Filter by label', ['(All)'] + labels, index=0)
    # Filter
    if sel_label == '(All)':
        view = ds
    else:
        view = [ex for ex in ds if ex['image_label'] == sel_label]

    if len(view) == 0:
        st.warning('No samples found for this filter.')
    else:
        idx = st.slider('Sample index', 0, len(view)-1, 0)
        ex = view[idx]
        img = fetch_image(ex['image_uri'])
        if img is not None:
            st.image(img, caption=f"{ex['image_label']}", use_container_width=True)
        else:
            st.error('Could not fetch image.')

        with st.expander('Metadata'):
            st.json({
                'image_label': ex['image_label'],
                'image_uri': ex['image_uri'],
                'image_issues': ex.get('image_issues', []),
                'object_labels': ex.get('object_labels', [])
            })

with tab3:
    st.header('📊 Simple stats')
    ds = hf_dataset()
    df = pd.DataFrame({'image_label': ds['image_label']})
    counts = df['image_label'].value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10,5))
    counts.head(30).plot(kind='bar', ax=ax)  # show top 30 for readability
    ax.set_title('Top 30 action labels')
    ax.set_ylabel('Count')
    ax.set_xlabel('image_label')
    st.pyplot(fig)
