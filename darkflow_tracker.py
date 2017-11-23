# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 15:40:20 2017

@author: Emil Wåreus
"""




from darkflow.net.build import TFNet
import cv2
import numpy as np
import time 


class Darkflow_subclass: 
    
    
    def __init__(self, threshold = 0.5):
        
        
        options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": threshold}
        self.tfnet = TFNet(options)
        self.label_size = 4
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        
    def predict_image(self, image, out_class = 'all'):
     
        
        result = self.tfnet.return_predict(image)
        
        if(out_class == 'all'):
            return result
        
        else:
           
            out_classes = [predict['label'] for predict in result]
            
            result_class = []
            for i, predicted_class in enumerate(out_classes):
                if(predicted_class == out_class):
                    result_class.append(result[i])
    
            return result_class
        
    def get_boxes(self, pred):
        out_boxes = [(predict['topleft']['y'],predict['topleft']['x'], predict['bottomright']['y'],predict['bottomright']['x']) for predict in pred]
        labels = ['{} {:.2f}'.format(predict['label'], predict['confidence']) for predict in pred] 
        return out_boxes, labels
        
    def draw_boxes(self,frame , pred):
        image = frame.copy()
        out_boxes = [(predict['topleft']['y'],predict['topleft']['x'], predict['bottomright']['y'],predict['bottomright']['x']) for predict in pred]
        out_classes = [predict['label'] for predict in pred]
        out_scores = [predict['confidence'] for predict in pred]
       
        #Make Thicker boxes
        thickness = (image.shape[1] + image.shape[0]) // 300
        for i, predicted_class in enumerate(out_classes):
            
            box = out_boxes[i]
            score = out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)

            top, left, bottom, right = box
            
          
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))
            
            
            #Put label on a smart place
            if top - self.label_size >= 0:
                text_origin = np.array([left, top - self.label_size])
            else:
                text_origin = np.array([left, top + 1])

            
            cv2.putText(image, label , (text_origin[0],text_origin[1]), self.font, 0.8, (0,255,0),2,cv2.LINE_AA)
           
            #Make Thicker rectangles
            for i in range(thickness):
                cv2.rectangle(image, (left + i, top + i ), 
                              (right - i, bottom - i), 
                              color = (0,255,0))
                
        return image
    
if __name__ == "__main__":
    
    tracker = Darkflow_subclass()
   

    
    while(True):    
        image = cv2.imread("./models/research/object_detection/PicturesTest/HM0.jpg")
    
        pred = tracker.predict_image(image, out_class = 'person')
        
        image = tracker.draw_boxes(image, pred)
        
        print(tracker.get_boxes(pred))
        image = cv2.resize(image, (720, 480))
        cv2.imshow('Frame',image)
        k = cv2.waitKey()
        if k == 27:
            break
 
    
    
    
    
    
    