import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# 学習済みモデルを読み込む
model = tf.keras.models.load_model('model.h5')

st.title("数字当てゲーム")
uploaded = st.file_uploader("👇ここに数字の絵をアップしてね", type=['png','jpg'])
if uploaded:
    img = Image.open(uploaded).convert('L').resize((28,28))
    st.image(img, caption='あなたのアップロード')
    x = np.array(img)/255.0
    x = x.reshape(1,28,28,1)
    pred = model.predict(x).argmax()
    st.success(f"コンピュータの予想は…『{pred}』です！")
