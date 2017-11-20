import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import webbrowser as wb
import tkinter as tk
import cv2

from collections import defaultdict
from io import StringIO
#from matplotlib import pyplot as plt
from PIL import Image
from collections import namedtuple
from PIL import ImageTk
from Tracker import Tracker
from IPython import embed
sys.path.append("..")


from keras.preprocessing import image
import numpy as np
from IPython import embed

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
import numpy as np
import h5py

from sklearn.decomposition import PCA

class MyPca():

    def __init__(self,nbrOFeatures):
        self.pca=PCA(n_components=nbrOFeatures)

    def flattenInput(self,x):
        if(x.shape!=2):
           print("problem with the input!")
        pass

    def fitPca(self,X):
        self.pca.fit(x)

    def transformPca(self,y):
        return (self.pca.transform(y))

class Extractor():
    def __init__ (self):
        base_model = InceptionV3(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('mixed9_1').output)
        #self.model = VGG16(weights='imagenet', include_top=False)

    def loadAndPreprocess(self,input_path='cropped.jpg'):
        img_path = input_path
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return(x)
    def predict(self,input_path='cropped.jpg'):
        x=self.loadAndPreprocess(input_path)
        return self.model.predict(x)

class Similarity():

    def eucledianDistance(self,x,y):
        return (np.sqrt(np.sum(np.power(a-b,2) for a, b in zip(x, y))))
    def manhatttan(self,x,y):
        return(sum(np.abs(a-b)for a,b in zip(x,y)))
    def cosSimilarity(self,x,y):
        numerator=np.dot(x,y)
        denominator=np.linalg.norm(x)*np.linalg.norm(y)
        #print(np.abs(x))
        return (numerator/denominator)
    def chebyshev(self,x,y,root=3):
        return(np.power(np.sum(np.power(a-b,2) for a,b in zip(x,y)),1/root))

    def getAllSimilarities(self,x,y):
        similarity=[self.eucledianDistance(x,y),self.manhatttan(x,y),self.cosSimilarity(x,y),self.chebyshev(x,y)]
        return (similarity)


class BuildBase():
        pathList=[]
        model=Extractor()
        measureSimilary=Similarity()

        def setImagePathCloths(self,inputList):
            self.pathList=inputList
        def setImageExtracted(self,image):
            self.imageExtracted=image
        def getSimilaririesForList(self):
            distance=[]
            vectorsCloths=[]
            for images in self.pathList:
                print(images)
                vectorsCloths.append(self.model.predict(images).flatten())
            for images in self.imageExtracted:
                vectorImage=[self.model.predict(images).flatten()]
            for index in range(0,len(vectorsCloths)):
                distance.append(self.measureSimilary.getAllSimilarities(vectorImage[0],vectorsCloths[index]))
            return (distance)


class Interactive ():
    # What model to download.
    MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
    NUM_CLASSES = 90
    

    def __init__(self):
        self.loadModel()

    def loadModel(self):
        self.tracker = Tracker()

    def load_image_into_numpy_array(self,image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


    def getBox(self,boxes,scores,classes,width,height):
        myList=[]
        for i in range(len(boxes)):
            if(classes[i]==0 and scores[i] > 0.5):
                # is this order correct i dont know......

                x0=boxes[i][1]
                y0=boxes[i][0]
                x1=boxes[i][3]
                y1=boxes[i][2]
                myList.append([x0,y0,x1,y1,classes[i]])
        return(myList)

    def objectDetection(self,TEST_IMAGE_PATH):
        result_list=[]
        self.tracker.start_model()
        image=cv2.imread(TEST_IMAGE_PATH[0])
        height, width = image.shape[:2]
        image, out_boxes, out_scores, out_classes = self.tracker.get_single_image(image)
        result_list.append(self.getBox(out_boxes, out_scores,out_classes,width,height))
        return (result_list)  



Rect = namedtuple('Rect', 'x0, y0, x1, y1')

#def cropImage2(crop_area,imagePath=TEST_IMAGE_PATH[0]):
#    image=Image.open(imagePath)
#    crop_area=crop_area
#    image.show()
#    print(crop_area)
#    image_cropped = image.crop(crop_area)
#    print("here")
#    image_cropped.show()

class ImageMapper(object):
    def __init__(self, image, img_rects):
        self.width, self.height = image.width(), image.height()
        #print(self.width, self.height)
        self.img_rects = img_rects
        
    def find_rect(self, x, y):
        #print("The cordinates is x:{} and y:{}".format(x,y))
        for i, r in enumerate(self.img_rects):
            #print("inne 1")
            #print(r)
            #print(" Are {} <= {} <= {} and{} <= {} <= {}".format(r.x0,x,r.x1,r.y0,y,r.y1))
            if (r.x0 <= x <= r.x1) and (r.y0 <= y <= r.y1):
                #print("inne 2")
                #print("The values are r.x0: {}, r.x1:{},r.y0:{} and r.y1:{}".format(r.x0,r.x1,r.y0,r.y1))
                return i
        #print("no click")
        return None

class Demo(tk.Frame):
    image_reacts=[]
    Rect = namedtuple('Rect', 'x0, y0, x1, y1')
    labels=[]
    PATH_TO_TEST_IMAGES_DIR = 'PicturesTest'
    TEST_IMAGE_PATH = [ os.path.join('PicturesTest', 'HM{}.jpg'.format(i)) for i in range(3, 4) ]

    def __init__(self, reacts,labels,master=None,):
        tk.Frame.__init__(self, master)
        self.grid()
        self.create_widgets()
        self.setReact(reacts)
        self.setLabels(labels)
        self.myPathList=['PicturesTest2/HM1.jpg','PicturesTest2/HM2.jpg','PicturesTest2/HM3.jpg','PicturesTest2/HM4.jpg','PicturesTest2/HM5.jpg','PicturesTest2/HM6.jpg','PicturesTest2/HM7.jpg','PicturesTest2/HM8.jpg','PicturesTest2/HM9.jpg','PicturesTest2/HM10.jpg','PicturesTest2/HM11.jpg','PicturesTest2/HM12.jpg']
        self.similarityModel=BuildBase()
        self.similarityModel.setImagePathCloths(self.myPathList)
    
    def setReact(self,image_rects):
        for image_react in image_rects:
            self.image_reacts.append(image_react)
        
    def create_widgets(self):
        self.msg_text = tk.StringVar()
        self.msg = tk.Message(self, textvariable=self.msg_text, width=100)
        self.msg.grid(row=0, column=0)

        #self.picture = tk.PhotoImage(file='image1.gif')
                        # 'x0, y0, x1, y1'
        #img_rects = [Rect(24, 24, 326, 548),
                     #Rect(401, 63, 996, 609)]
        #self.imagemapper = ImageMapper(self.picture, img_rects)
        path = TEST_IMAGE_PATH[0]

        #Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
        self.picture = ImageTk.PhotoImage(Image.open(path))

        self.imagemapper = ImageMapper(self.picture, self.image_reacts)
        # use Label widget to display image
        self.image = tk.Label(self, image=self.picture, borderwidth=0)
        self.image.bind('<Button-1>', self.image_click)
        self.image.grid(row=1, column=0)

        self.quitButton = tk.Button(self, text='Quit', command=self.quit)
        self.quitButton.grid(row=2, column=0)

    def image_click(self, event):
        hit = self.imagemapper.find_rect(event.x, event.y)
        self.msg_text.set('{} clicked'.format('nothing' if hit is None else
                                              'rect[{}]'.format(hit)))
        if(hit!=None):
            #print("the label of the hit is {}".format(self.labels[hit]))
            num=self.labels[hit]
            image_object=self.image_reacts[hit]
            crop_area= (image_object.x0,image_object.y0,image_object.x1,image_object.y1)
            self.cropImage(crop_area,self.TEST_IMAGE_PATH[0])
    
    def setLabels(self,labels):
        self.labels=labels
           
    def openLink(self, Url):
        #print(Url)
        wb.open_new_tab(Url)
        
    def cropImage(self, crop_area,imagePath):
        img=Image.open(imagePath)
        #crop_area=crop_area
        crop_area=crop_area
        image_cropped = img.crop(crop_area)
        width, height = img.size
        #print(img.size)
        #print("here")
        #image_cropped.show()
        image_cropped.save("PicturesTest/out.jpg", "JPEG")
        EXTRACTED_IMAGE_PATH = ['PicturesTest/out.jpg']
        self.similarityModel.setImageExtracted(EXTRACTED_IMAGE_PATH)
        #self.nicePrintOut(self.similarityModel.getSimilaririesForList())
        minValue,index=self.getTheMostSimilar(self.similarityModel.getSimilaririesForList())
        link=self.getTheCorrectLink(index)
        self.openLink(link)
        

    def getTheMostSimilar(self,similarityList):
        minValue=float('inf')
        index=None
        for indexNbr,cases in enumerate(similarityList):
            print("The image is:{} and the distances is:{}".format(indexNbr+1,cases))
            if cases[1] <minValue:
                index=indexNbr
                minValue=cases[1]
        print(minValue,index)
        return (minValue,index)

    def getTheCorrectLink(self,SimilarityIndex):
        Links=['http://files.hm.com/share/ffc/9pc-star-4nf47cmauszyjf4/resources/img/collection/desktop/erdem-x-hm-designer-collaboration-products-men-5.jpg',
                'http://files.hm.com/share/ffc/9pc-star-4nf47cmauszyjf4/resources/img/collection/desktop/erdem-x-hm-designer-collaboration-products-ladies-29.jpg',
                'http://files.hm.com/share/ffc/9pc-star-4nf47cmauszyjf4/resources/img/collection/desktop/erdem-x-hm-designer-collaboration-products-ladies-27.jpg',
                'http://files.hm.com/share/ffc/9pc-star-4nf47cmauszyjf4/resources/img/collection/desktop/erdem-x-hm-designer-collaboration-products-ladies-30.jpg',
                'http://files.hm.com/share/ffc/9pc-star-4nf47cmauszyjf4/resources/img/collection/desktop/erdem-x-hm-designer-collaboration-products-men-14.jpg',
                'http://files.hm.com/share/ffc/9pc-star-4nf47cmauszyjf4/resources/img/collection/desktop/erdem-x-hm-designer-collaboration-products-ladies-28.jpg',
                'http://files.hm.com/share/ffc/9pc-star-4nf47cmauszyjf4/resources/img/collection/desktop/erdem-x-hm-designer-collaboration-products-ladies-27.jpg',
                'http://files.hm.com/share/ffc/9pc-star-4nf47cmauszyjf4/resources/img/collection/desktop/erdem-x-hm-designer-collaboration-products-ladies-31.jpg',
                'http://files.hm.com/share/ffc/9pc-star-4nf47cmauszyjf4/resources/img/collection/desktop/erdem-x-hm-designer-collaboration-products-ladies-30.jpg',
                'http://files.hm.com/share/ffc/9pc-star-4nf47cmauszyjf4/resources/img/collection/desktop/portrait/erdem-x-hm-designer-collaboration-products-men-16.jpg',
                'http://files.hm.com/share/ffc/9pc-star-4nf47cmauszyjf4/resources/img/collection/desktop/erdem-x-hm-designer-collaboration-products-men-10.jpg',
                'http://files.hm.com/share/ffc/9pc-star-4nf47cmauszyjf4/resources/img/collection/desktop/erdem-x-hm-designer-collaboration-products-ladies-17.jpg']
        print(SimilarityIndex)
        print("here")
        #print(Links[SimilarityIndex])
        #print("here")
        return Links[SimilarityIndex]
        
    def openUrl(self,listsOfSimilarity):
        mydict=dict()
        for index,myList in enumerate(listsOfSimilarity):
            mydict[self.myPathList[index]]=(myList[2],'http://www.hm.com/se/erdem')
        self.openLink(mydict[max(mydict, key=mydict.get)][1])
        
        
PATH_TO_TEST_IMAGES_DIR = 'PicturesTest'
TEST_IMAGE_PATH = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'HM{}.jpg'.format(i)) for i in range(3, 4) ]
model = Interactive()
results = model.objectDetection(TEST_IMAGE_PATH)
results = np.squeeze(results)
Reacts = []
Labels = []
try:
    results.shape[1]
    for result in results:
        Reacts.append(Rect(result[0], result[1],result[2], result[3]))
        Labels.append(result[4])
except (ValueError,IndexError):
    Reacts.append(Rect(results[0], results[1],results[2], results[3]))
    Labels.append(results[4])
app = Demo(Reacts,Labels)
app.master.title('Image Mapper')
app.mainloop()


