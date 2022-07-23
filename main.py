import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow.python.keras import utils 
import streamlit as st



def main():
    st.title("""
    IMAGE SUPER RESOLUTION USING CONVULSION NEURAL NETWORK
    """)
    st.text("""""")
    img_path_list=["D:\pythonCode\projects\multimediaproject\APP\static\images\image1.png"] 
    image=Image.open(img_path_list[0])
    st.image(image,width=600)
    st.write("""
    # Welcome to the Image Super Resolution App
    This app is designed to help you to upsample your images.
    """)
    st.write("""
    you have to upload your own test images to test it!!!
    """)
    st.write("""
    # Upload your image
    """)
    uploaded_file = st.file_uploader("Upload your image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, width=300)
   
    if st.button("Predict"):
        st.write("""
        # Predicting your image
        """)
        model_path='D:\pythonCode\projects\multimediaproject\APP\static\model\model1.h5'
        model=load_model(model_path)
        image=Image.open(uploaded_file)
        image=img_to_array(image)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image=cv2.resize(image,(256,256))
        image=image.astype('float32')/255.0
        image=img_to_array(image)
        image=np.expand_dims(image,axis=0)
        prediction=model.predict(image)
        st.image(prediction[0],width=300,clamp=True,channels='BGR')
        st.write("""
        # Your image is ready
        """)
        st.write("""
         Thank you for using this app
        """)
        st.write("""
         Hope you like it
        """)
        st.write("""
         Created By:
        """)
        st.write("""
         Charan K S
        """)
        

    

main()