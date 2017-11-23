#from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
#from keras.applications.vgg16 import preprocess_input
import numpy as np


from BuildingBlocks import AbstracExtractor
from IPython import embed
from keras.applications.inception_v3 import InceptionV3

from keras.applications.xception import Xception

from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
import numpy as np
import h5py
import indicoio

from PIL import Image, ImageTk
from IPython import embed
import os

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

class ExtractorDeepLearninCNN(AbstracExtractor):
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


class ExtractorHistogram(AbstracExtractor):

    def __init__(self):
        dataSetDicrector='PicturesTest2'

    def loadAndPreprocess(self,input_path):
        img = image.load_img(input_path, target_size=(500, 500))
        search_histo = np.array(img.histogram())
        return(search_histo)

    def predict(self,input_path):
        x=self.loadAndPreprocess(input_path)
        return (x)

class ExtractorXception(AbstracExtractor):

    def __init__(self):
            self.model=Xception(weights='imagenet', include_top=False)

    def loadAndPreprocess(self,input_path):
        img_path = input_path
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return(x)

    def predict(self,input_path):
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
        #print(similarity)
        return (similarity)


class BuildBase():

        def __init__(self,modelSetup):
            print(modelSetup)
            self.pathList=[]
            if modelSetup =="Deep":
                self.model = ExtractorDeepLearninCNN()
            if modelSetup =="Deep2":
                self.model = ExtractorXception()
            if modelSetup == "Hist" :
                self.model=ExtractorHistogram()
            self.measureSimilary=Similarity()

        def setImagePathCloths(self,inputList):
            self.pathList=inputList
        def setImageExtracted(self,image):
            self.imageExtracted=image
        def getSimilaririesForList(self):

            distance=[]
            vectorsCloths=[]
            for images in self.pathList:
                vectorsCloths.append(self.model.predict(images).flatten())
                #print(images)
            for images in self.imageExtracted:
                #print(images)
                vectorImage=[self.model.predict(images).flatten()]
            for index in range(0,len(vectorsCloths)):
                distance.append(self.measureSimilary.getAllSimilarities(vectorImage[0],vectorsCloths[index]))
                #print(self.measureSimilary.getAllSimilarities(vectorImage[0],vectorsCloths[index]))
            return (distance)

#myPathList=['PicturesTest/HM1.jpg','PicturesTest/HM2.jpg','PicturesTest/HM3.jpg','PicturesTest/HM4.jpg','PicturesTest/HM5.jpg','PicturesTest/HM6.jpg']
#myPathList=['PicturesTest/HMTest1.jpg','PicturesTest/HMTest2.jpg','PicturesTest/HMTest3.jpg','PicturesTest/HMTest4.jpg','PicturesTest/HM2.jpg','PicturesTest/HM4.jpg','PicturesTest/HM1.jpg']
#test=BuildBase()
#test.setImagePathCloths(myPathList)
#embed()
print("done")

