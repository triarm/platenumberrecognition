from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import base64
from io import BytesIO
import json, os, cv2, shutil
import numpy as np

# Keras
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
SEGMENTATION_MODEL_PATH = 'SegmentationModel/UNET_VGG19/model.h5'
CHARACTER_MODEL_PATH = 'Character Recognition Model/model_vgg16/model.h5'

def BinaryIoU_func(y_true, y_pred):
        """
        Function to calculate IoU
        """
        pred = tf.where(y_pred>=0.5, 1, 0)
        pred = tf.cast(pred, dtype = tf.float32)
        true = tf.cast(y_true, dtype = tf.float32)
        n_true = tf.reduce_sum(true)
        n_pred = tf.reduce_sum(pred)
        intersection = tf.reduce_sum(pred * true)
        union = n_true + n_pred - intersection
        iou = intersection/union
        return iou

class ANPR:
    def __init__(self, segmentation_model_path, character_model_path):
        """
        Load model
        """
        self.seg_model = load_model(segmentation_model_path, custom_objects = {"BinaryIoU_func" : self.BinaryIoU_func})
        self.char_model = load_model(character_model_path)

    def BinaryIoU_func(y_true, y_pred):
        """
        Function to calculate IoU
        """
        pred = tf.where(y_pred>=0.5, 1, 0)
        pred = tf.cast(pred, dtype = tf.float32)
        true = tf.cast(y_true, dtype = tf.float32)
        n_true = tf.reduce_sum(true)
        n_pred = tf.reduce_sum(pred)
        intersection = tf.reduce_sum(pred * true)
        union = n_true + n_pred - intersection
        iou = intersection/union
        return iou

    def open_image(self, image_path, size = (256,256)):
        """
        Function to convert image to array
        """
        image = Image.open(image_path).resize(size).convert('RGB')
        image_array = np.array(image)
        return image_array

    def unet_predict(self, image_path):
        """
        Function to predict the segmentation
        """
        image_array = self.open_image(image_path)
        image_array = image_array[np.newaxis, ...]
        pred = self.seg_model.predict(image_array)[0]
        pred = np.where(pred >= 0.5, 255, 0)[:,:,0]

        return pred.astype(np.uint8)

    def postprocess_segmentation(self, unet_result, image_path):
        """
        Function to crop the image from segmentation model
        """
        src1_mask = cv2.resize(unet_result, (5*256, 5*256), interpolation = cv2.INTER_LANCZOS4)
        src1 = cv2.resize(cv2.imread(image_path), (5*256,5*256))
        src1_mask = cv2.cvtColor(src1_mask,cv2.COLOR_GRAY2BGR)#change mask to a 3 channel image 
        mask_out = cv2.subtract(src1_mask,src1)
        cv2_image = cv2.subtract(src1_mask,mask_out)

        gray = cv2.cvtColor(cv2_image,cv2.COLOR_BGR2GRAY)
        try:
            contours,hierarchy = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            cnt = contours[0]
            x,y,w,h = cv2.boundingRect(cnt)
        except IndexError:
            print(f"Model can't detect plate")

        crop = cv2_image[y:y+h,x:x+w]
        h,w,_ = crop.shape
        scaler = 400//h
        crop = cv2.resize(crop, (300,300))

        return crop

    def seg_prediction(self, image_path):
        """
        Function to localize plate
        """
        result = self.unet_predict(image_path)
        postprocess = self.postprocess_segmentation(unet_result = result,
                                               image_path = image_path)
        
        return postprocess

    def char_predict(self, img):
        """
        Function to predict character from image
        """
        mapping = {}
        CHARACTER = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for idx, c in enumerate(CHARACTER):
            mapping[idx] = c

        image_array = np.array(Image.fromarray(img).resize((75,75)).convert('RGB'))
        image_array = image_array[np.newaxis, ...]
        prediction = self.char_model.predict(image_array)
        idx = np.argmax(prediction)
        return mapping[idx]
    
    def sort_contours(self, contours):
        """
        Sorting the contours from the left box
        """
        # construct the list of bounding boxes and sort them from top to bottom
        boundingBoxes = [cv2.boundingRect(c) for c in contours]
        (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes)
        , key=lambda b: b[1][0], reverse=False))
        # return the list of sorted contours
        return contours

    def identify_char(self, crop_img):
        """
        Function to segment character and identify it
        """
        list_char = ''
        image = crop_img.copy()
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        kernel_erosion = np.ones((3,3), np.uint8)
        kernel_dilation = np.ones((3,3), np.uint8)
        img_erosion = cv2.erode(thresh,kernel_erosion,iterations = 1)
        img_dilation = cv2.dilate(img_erosion, kernel_dilation, iterations=2)
        #find contours
        ctrs, hier = cv2.findContours(img_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        hierarchy = hier[0]

        inner_contours = [c[0] for c in zip(ctrs, hierarchy) if c[1][3] == -1]

        sorted_contours = self.sort_contours(inner_contours)
        # #sort contours
        # # sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

        area = list(map(lambda x: cv2.contourArea(x), sorted_contours))
        threshold_area = np.quantile(area, 0.25)

        for i, ctr in enumerate(sorted_contours):
             #if hier[0,i,3] == -1:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(ctr)

                if (w > h) or (area[i] < 300) or (area[i] > 4100):
                    continue

                calon_img = image[y:y+h, x:x+w, :]
                class_char = self.char_predict(calon_img)
                cv2.rectangle(image,(x,y),( x + w, y + h ),(0,255,0),2)
                cv2.putText(image, f'{class_char}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 1)
                list_char += class_char

        return {'images': image, 'character': list_char}

    def predict_all(self, image_path):
        """
        Function to predict from vehicle image to character
        """
        seg_prediction = self.seg_prediction(image_path)
        OUTPUT = self.identify_char(seg_prediction)

        return OUTPUT

anpr = ANPR('./SegmentationModel/UNET_VGG19/model.h5', './Character Recognition Model/model_vgg16/model.h5')
print('Model loaded. Check http://127.0.0.1:5000/')


# def model_predict(img_path, model):
#     img = image.load_img(img_path, target_size=(224, 224))

#     # Preprocessing the image
#     x = image.img_to_array(img)
#     # x = np.true_divide(x, 255)
#     x = np.expand_dims(x, axis=0)

#     # Be careful how your trained model deals with the input
#     # otherwise, it won't make correct prediction!
#     x = preprocess_input(x, mode='caffe')

#     preds = model.predict(x)
#     return preds


@app.route('/', methods=['GET', 'POST'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/submit', methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        # Get the file from post request
        img = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(img.filename))
        img.save(file_path)

        # Make prediction
        preds = anpr.predict_all(file_path)

        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        result_img = Image.fromarray(preds['images'], mode = "RGB")
        figfile = BytesIO()
        result_img.save(figfile, format='png')
        figfile.seek(0)
        figdata_png = base64.b64encode(figfile.getvalue())
        pred_result = str(figdata_png)[2:-1]
        result_char = preds['character']
        return render_template("index.html", prediction = result_char,
                                img_path = pred_result)
    return None


if __name__ == '__main__':
    app.run(debug=True)

