import pandas as pd
import numpy as np
import streamlit as st 
from skimage import io

st.title(':violet[Сжатие фото] :camera:')
st.subheader('                              ')
uploaded_image = st.file_uploader(':violet[Загрузите ваше изображение (обработка ч/б)]', type=['png','jpeg','jpg'])

if uploaded_image is not None:
    image_np = io.imread(uploaded_image)[:, :, 0]
else:
    st.stop()
    
try:
    with st.form(key='my_form'):
        submit_button = st.form_submit_button(label='Отправить')
        top_k=int(st.text_input(label='Напишите меру сжатия (в сингулярных числах)'))   
except:
    st.stop()
        
st.image(image_np, caption='Исходное')
U, sing_values, V = np.linalg.svd(image_np)
sigma = np.zeros(shape=image_np.shape)
np.fill_diagonal(sigma, sing_values)
tr_U = U[:, :top_k]
tr_sigma = sigma[:top_k, :top_k]
tr_V = V[:top_k, :]
final = tr_U@tr_sigma@tr_V
final_scaled = final - np.min(final)
final_scaled /= np.max(final_scaled) 
final_image = (final_scaled * 255).astype(np.uint8)
st.image(final_image, caption='После сжатия') 
