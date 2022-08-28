import streamlit as st
import tensorflow as tf
import numpy as np
import urllib.request
import cv2
import streamlit.components.v1 as components

st.cache()
def download_model():
    url = 'https://github.com/mazy06000/topic-modeling-image-labeling/releases/download/model/image_labeling_model.h5'
    filename = url.split('/')[-1]
    urllib.request.urlretrieve(url, filename)

if "model" not in st.session_state:
    # download_model()
    st.session_state['model'] = tf.keras.models.load_model("image_labeling_model.h5")

CLASSES = ['drink', 'food', 'interior', 'menu', 'outside']

header = st.container()
uploading = st.container()
characteristic = st.container()

with header:
    st.markdown("""<div style="display:flex;justify-content:center;"><h1>IMAGE LABELING</h1></div>""", unsafe_allow_html=True)

with uploading:

    image_uploaded = st.file_uploader("Choose an image between: drink, food, interior, menu, outside")

    if image_uploaded is not None:

        nparr = np.fromstring(image_uploaded.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_to_predict = cv2.resize(img, (224, 224))
        img_to_predict = np.expand_dims(img_to_predict, axis=0)

        y_pred = st.session_state['model'].predict(img_to_predict)
        y_class = CLASSES[y_pred.argmax(axis=1)[0]]

        _, center, _ = st.columns([0.5,2,0.5])
        center.image(img, channels="BGR", use_column_width="always")
        components.html(f"""
            <div style="display:flex; flex-direction: column;align-items: center;">
                <p>It is an image of</p>
                <h1 style="color:green">{y_class.upper()}</h1>
            </div>
        """)

with characteristic:
    st.header("Characteristic")
    left, right = st.columns(2)
    left.subheader("About Model")
    left.markdown('<div><b>Model:</b> VGG16 by Transfer Learning</div>',
                  unsafe_allow_html=True)

    right.subheader("Model performance")
    right.markdown('<div><b>Test Accuracy:</b> 90%</div>', unsafe_allow_html=True)