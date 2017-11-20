import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import webbrowser as wb
import tkinter as tk

from collections import defaultdict
from io import StringIO
#from matplotlib import pyplot as plt
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util
from collections import namedtuple
from PIL import ImageTk
sys.path.append("..")





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
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    def __init__(self):
        self.loadModel()
    def loadModel(self):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
    def load_image_into_numpy_array(self,image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


    def getBox(self,boxes,scores,classes,width,height):
        boxes=np.squeeze(boxes)
        scores=np.squeeze(scores)
        classes=np.squeeze(classes).astype(np.int32)
        myList=[]
        for index,score in enumerate(scores):
            if score > 0.5:
                #print("here")
                x0=boxes[index][1]*width
                y0=boxes[index][0]*height
                x1=boxes[index][3]*width
                y1=boxes[index][2]*height
                myList.append([x0,y0,x1,y1,classes[index]])
        return(myList)

    def objectDetection(self,TEST_IMAGE_PATH):
                    
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        self.category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)
                    #getBox(boxes, scores,classes,width,height)
                    # The function aboe print the boxes and so on! 
                    # 
                    #plt.figure(figsize=IMAGE_SIZE)
                    #plt.imshow(image_np)
                    result_list.append(self.getBox(boxes, scores,classes,width,height))
                return result_list




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
        #print(x,y)
        for i, r in enumerate(self.img_rects):
            if (r.x0 <= x <= r.x1) and (r.y0 <= y <= r.y1):
                return i
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
        self.myPathList=['PicturesTest/HM0.jpg','PicturesTest/HM1.jpg','PicturesTest/HM2.jpg','PicturesTest/HM3.jpg','PicturesTest/HM4.jpg']
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
        #print("the label of the hit is {}".format(self.labels[hit]))
        num=self.labels[hit]
        image_object=self.image_reacts[hit]
        crop_area= (image_object.x0,image_object.y0,image_object.x1,image_object.y1)
        self.cropImage(crop_area,self.TEST_IMAGE_PATH[0])
        #self.openLink(num)
    
    def setLabels(self,labels):
        self.labels=labels
           
    def openLink(self, Url):
        print(Url)
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
        self.openUrl(self.similarityModel.getSimilaririesForList())
        
    #def nicePrintOut(self,listsOfSimilarity):
    #    for list in listsOfSimilarity:
    #        print(list)
    #        print('/n')
        
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


