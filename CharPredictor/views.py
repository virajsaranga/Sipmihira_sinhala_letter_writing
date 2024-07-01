from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from keras.models import model_from_json

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import io
from PIL import Image
import json

from .models_dir.objectdetection import ObjectDetection

# Create your views here.
from rest_framework.decorators import api_view
from rest_framework.response import Response

def config():
    
    json_file = open(os.path.join(os.getcwd(), "CharPredictor/models_dir/char_model.json"), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    char_model = model_from_json(loaded_model_json)
    char_model.load_weights(os.path.join(os.getcwd(), "CharPredictor/models_dir/char_model.h5"))
    print("Loaded model from disk")
    
    obj_model = ObjectDetection()
    
    return char_model, obj_model


classes = ["අ", "උ", "ග", "ට", "ත", "ප", "ද", "ර", "ල", "ස", "ඉ", "ය", "ම"]

def preprocess(image):
    image = cv2.resize(image, (224, 224))
    _, image = cv2.threshold(image,120, 255, cv2.THRESH_BINARY_INV)
    
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # image = cv2.dilate(image, kernel, iterations=15)
    
    return image

def process_result(result, topK):
    
    ind = np.argpartition(result[0], -topK)[-topK:]
    
    prob = result[0][ind]
    op = []
    for j in ind:
        op.append(classes[j])

    return prob, op


def predict(image, original):
    
    char_model, obj_model = config()
    image = np.array(image)
    objects = obj_model.get_objects(image)
    
    if len(objects) >= 3 or len(objects) == 0:
        obj = {
            "No.of Objects": len(objects),
            "Status": "එය වැරදි",
            "Char": "-",
            "Understanding": "හදුනාගැනීම දුර්වලයි",
            "Clarity": "ලිවිම දුර්වලයි"
        }

        
    elif len(objects) == 2:
        
        image1 = preprocess(objects[0])
        # if len(image1.shape) == 2:
        #     image1 = np.dstack([image1]*3)
        x1 = image1 / 255.0
    
        result1 = char_model.predict(np.expand_dims(x1, axis=0))
        prob1, op1 = process_result(result1, 5)
        # class1 = classes[np.argmax(result1)]
        # prob1 = np.max(result1)
        
        image2 = preprocess(objects[1])
        # if len(image2.shape) == 2:
        #     image2 = np.dstack([image2]*3)
        x2 = image2 / 255.0
        
        result2 = char_model.predict(np.expand_dims(x2, axis=0))
        prob2, op2 = process_result(result2, 5)
        # class2 = classes[np.argmax(result2)]
        # prob2 = np.max(result2)
        
        # print(result1, prob1)
        # cv2.imshow("Image", image1)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows() # destroys the window showing image
        
        # print(result2, prob2)
        # cv2.imshow("Image", image2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows() # destroys the window showing image
        
        # if class1 == class2:
        #     if prob1 > prob2:
        #         prob = prob1
        #     else:
        #         prob = prob2

        #     if prob >= 0.8:
        #         clarity = "Excellent"
        #     elif prob >= 0.5:
        #         clarity = "Good"
        #     else:
        #         clarity = "Improvement Needed"
            
        #     obj = {
        #         "No.of Objects": 2,
        #         "Status": "Correct",
        #         "Char": class1,
        #         "Understanding": "Both objects are correct but asked to write 1",
        #         "Clarity": clarity
        #     }
            
        if original in op1:
            idx = op1.index(original)
            prob = prob1[idx]
            
            if prob >= 0.8:
                clarity = "ඉතා හොදයි "
            elif prob >= 0.5:
                clarity = "හොදයි"
            else:
                clarity = "තව පුරුදුවෙමු"
            
            obj = {
                "No.of Objects": 2,
                "Status": "නිවැරදි",
                "Char": original,
                "Understanding": "එක් අකුරක් නිවරදි වන අතර ඔබ තවත් දෙයක් ඇද ඇත",
                "Clarity": clarity
            }
            
        elif original in op2:
            idx = op2.index(original)
            prob = prob2[idx]
            
            if prob >= 0.8:
                clarity = "ඉතා හොදයි"
            elif prob >= 0.5:
                clarity = "හොදයි"
            else:
                clarity = "තව පුරුදුවෙමු"
            
            obj = {
                "No.of Objects": 2,
                "Status": "නිවැරදි",
                "Char": original,
                "Understanding": "එක් අකුරක් නිවරදි වන අතර ඔබ තවත් දෙයක් ඇද ඇත",
                "Clarity": clarity
            }
            
        else:
            
            obj = {
                "No.of Objects": 2,
                "Status": "එය වැරදි ",
                "Char": "-",
                "Understanding": "හදුනාගැනීම දුර්වලයි ",
                "Clarity": "ලිවිම දුර්වලයි"
            }
            
    elif len(objects) == 1:
        
        image = preprocess(image)
        # if len(image.shape) == 2:
        #     image = np.dstack([image]*3)
        image = image / 255.0

        result = char_model.predict(np.expand_dims(image, axis=0))
        prob, op = process_result(result, 5)
        # class_prob = classes[np.argmax(result)]
        # prob = np.max(result)
        
        # cv2.imshow("Image", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows() # destroys the window showing image
        
        if original in op:
            idx = op.index(original)
            prob_s = prob[idx]
            
            if prob_s >= 0.8:
                clarity = "ඉතා හොදයි"
            elif prob_s >= 0.5:
                clarity = "හොදයි"
            else:
                clarity = "තව පුරුදුවෙමු"
            
            obj = {
                "No.of Objects": 1,
                "Status": "එය නිවැරදි",
                "Char": original,
                "Understanding": "හඳුනාගැනීම විශිෂ්ටයි" ,
                "Clarity": clarity
            }
        
        else:
            
            obj = {
                "No.of Objects": 1,
                "Status": "එය වැරදි",
                "Char": '-',
                "Understanding": "හදුනාගැනීම දුර්වලයි" ,
                "Clarity": 'ලිවිම දුර්වලයි'
            }

    return obj


@csrf_exempt
def request_video(request):
    
    return JsonResponse({"response": "OK"}, status=200)

@csrf_exempt
def submit_char(request):
    if request.method == 'POST':

        im_bytes = request.FILES['image'].file.read()
        image = Image.open(io.BytesIO(im_bytes))
        obj = predict(image, request.POST['original'])
        
        return JsonResponse(obj, status=200, safe=False)





