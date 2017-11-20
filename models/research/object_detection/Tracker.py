# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 10:22:12 2017

@author: Emil
"""

#keras imports
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
import keras.backend as K
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda, GlobalAveragePooling2D



import numpy as np
import colorsys
import random
import cv2
import io
from collections import defaultdict
import configparser
import time


#From Utils
from yad2k.models.keras_yolo import space_to_depth_x2, space_to_depth_x2_output_shape, yolo_eval, yolo_head
#from utils_b import WeightReader, decode_netout, draw_boxes



class Tracker:

    
    
    def __init__(self):
        
        self.LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.TRUE_BOX_BUFFER  = 50
        self.CLASS            = len(self.LABELS)
        self.SCORE_THRESHOLD    = 0.3#0.5
        self.IOU_THRESHOLD    = 0.5#0.45
        self.ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
        self.ANCHORS = np.array(self.ANCHORS).reshape(-1, 2)
        self.TRUE_BOX_BUFFER  = 50


        self.labels   = list(self.LABELS)
        self.nb_class = len(self.labels)
        self.nb_box   = 5
        self.class_wt = np.ones(self.nb_class, dtype='float32')

       

    def summary(self):
        self.model.summary()
    
        
    def unique_config_sections(self, config_file):
        """Convert all config sections to have unique names.
    
        Adds unique suffixes to config sections for compability with configparser.
        """
        section_counters = defaultdict(int)
        output_stream = io.StringIO()
        with open(config_file) as fin:
            for line in fin:
                if line.startswith('['):
                    section = line.strip().strip('[]')
                    _section = section + '_' + str(section_counters[section])
                    section_counters[section] += 1
                    line = line.replace(section, _section)
                output_stream.write(line)
        output_stream.seek(0)
        return output_stream
    
    
    def get_model(self):
        config_path = 'Config_Weights/yolo.cfg'
        weights_path = 'Config_Weights/yolo.weights'
        assert config_path.endswith('.cfg'), '{} is not a .cfg file'.format(
            config_path)
        assert weights_path.endswith(
            '.weights'), '{} is not a .weights file'.format(weights_path)
       
        # Load weights and config.
        #print('Loading weights.')
        weights_file = open(weights_path, 'rb')
        weights_header = np.ndarray(
            shape=(4, ), dtype='int32', buffer=weights_file.read(16))
        #print('Weights Header: ', weights_header)
        
        #print('Parsing Darknet config.')
        unique_config_file = self.unique_config_sections(config_path)
        cfg_parser = configparser.ConfigParser()
        cfg_parser.read_file(unique_config_file)
    
        #print('Creating Keras model.')
        
        image_height = int(cfg_parser['net_0']['height'])
        image_width = int(cfg_parser['net_0']['width'])
        prev_layer = Input(shape=(image_height, image_width, 3))
        all_layers = [prev_layer]
    
        weight_decay = float(cfg_parser['net_0']['decay']
                             ) if 'net_0' in cfg_parser.sections() else 5e-4
        count = 0
        for section in cfg_parser.sections():
            print('Parsing section {}'.format(section))
            if section.startswith('convolutional'):
                filters = int(cfg_parser[section]['filters'])
                size = int(cfg_parser[section]['size'])
                print(filters)
                stride = int(cfg_parser[section]['stride'])
                pad = int(cfg_parser[section]['pad'])
                activation = cfg_parser[section]['activation']
                batch_normalize = 'batch_normalize' in cfg_parser[section]
    
                # padding='same' is equivalent to Darknet pad=1
                padding = 'same' if pad == 1 else 'valid'
    
                # Setting weights.
                # Darknet serializes convolutional weights as:
                # [bias/beta, [gamma, mean, variance], conv_weights]
                prev_layer_shape = K.int_shape(prev_layer)
    
                # TODO: This assumes channel last dim_ordering.
                weights_shape = (size, size, prev_layer_shape[-1], filters)
                darknet_w_shape = (filters, weights_shape[2], size, size)
                weights_size = np.product(weights_shape)
    
                #print('conv2d', 'bn'
                #      if batch_normalize else '  ', activation, weights_shape)
    
                conv_bias = np.ndarray(
                    shape=(filters, ),
                    dtype='float32',
                    buffer=weights_file.read(filters * 4))
                count += filters
    
                if batch_normalize:
                    bn_weights = np.ndarray(
                        shape=(3, filters),
                        dtype='float32',
                        buffer=weights_file.read(filters * 12))
                    count += 3 * filters
    
                    # TODO: Keras BatchNormalization mistakenly refers to var
                    # as std.
                    bn_weight_list = [
                        bn_weights[0],  # scale gamma
                        conv_bias,  # shift beta
                        bn_weights[1],  # running mean
                        bn_weights[2]  # running var
                    ]
    
                conv_weights = np.ndarray(
                    shape=darknet_w_shape,
                    dtype='float32',
                    buffer=weights_file.read(weights_size * 4))
                count += weights_size
    
                # DarkNet conv_weights are serialized Caffe-style:
                # (out_dim, in_dim, height, width)
                # We would like to set these to Tensorflow order:
                # (height, width, in_dim, out_dim)
                # TODO: Add check for Theano dim ordering.
                conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
                conv_weights = [conv_weights] if batch_normalize else [
                    conv_weights, conv_bias
                ]
    
                # Handle activation.
                act_fn = None
                if activation == 'leaky':
                    pass  # Add advanced activation later.
                elif activation != 'linear':
                    raise ValueError(
                        'Unknown activation function `{}` in section {}'.format(
                            activation, section))
    
                # Create Conv2D layer
                conv_layer = (Conv2D(
                    filters, (size, size),
                    strides=(stride, stride),
                    kernel_regularizer=l2(weight_decay),
                    use_bias=not batch_normalize,
                    weights=conv_weights,
                    activation=act_fn,
                    padding=padding))(prev_layer)
    
                if batch_normalize:
                    conv_layer = (BatchNormalization(
                        weights=bn_weight_list))(conv_layer)
                prev_layer = conv_layer
    
                if activation == 'linear':
                    all_layers.append(prev_layer)
                elif activation == 'leaky':
                    act_layer = LeakyReLU(alpha=0.1)(prev_layer)
                    prev_layer = act_layer
                    all_layers.append(act_layer)
    
            elif section.startswith('maxpool'):
                size = int(cfg_parser[section]['size'])
                stride = int(cfg_parser[section]['stride'])
                all_layers.append(
                    MaxPooling2D(
                        padding='same',
                        pool_size=(size, size),
                        strides=(stride, stride))(prev_layer))
                prev_layer = all_layers[-1]
    
            elif section.startswith('avgpool'):
                if cfg_parser.items(section) != []:
                    raise ValueError('{} with params unsupported.'.format(section))
                all_layers.append(GlobalAveragePooling2D()(prev_layer))
                prev_layer = all_layers[-1]
    
            elif section.startswith('route'):
                ids = [int(i) for i in cfg_parser[section]['layers'].split(',')]
                layers = [all_layers[i] for i in ids]
                if len(layers) > 1:
                    print('Concatenating route layers:', layers)
                    concatenate_layer = concatenate(layers)
                    all_layers.append(concatenate_layer)
                    prev_layer = concatenate_layer
                else:
                    skip_layer = layers[0]  # only one layer to route
                    all_layers.append(skip_layer)
                    prev_layer = skip_layer
    
            elif section.startswith('reorg'):
                block_size = int(cfg_parser[section]['stride'])
                assert block_size == 2, 'Only reorg with stride 2 supported.'
                all_layers.append(
                    Lambda(
                        space_to_depth_x2,
                        output_shape=space_to_depth_x2_output_shape,
                        name='space_to_depth_x2')(prev_layer))
                prev_layer = all_layers[-1]
    
            elif section.startswith('region'):
                    with open('{}_anchors.txt'.format('Output'), 'w') as f:
                        print(f)
                        
                        #print(cfg_parser[section]['anchors'], file=f)
        
            elif (section.startswith('net') or section.startswith('cost') or
                  section.startswith('softmax')):
                pass  # Configs not currently handled during model definition.
    
            else:
                raise ValueError(
                    'Unsupported section header type: {}'.format(section))
    
        # Create and save model.
        self.model = Model(inputs=all_layers[0], outputs=all_layers[-1])
        return self.model
    
    def start_model(self):
        '''
        This needs to run to be able to use 
        get_single_image
        
        '''
        #Make the model
        self.yolo_model = self.get_model()
        self.summary()
        
        self.sess = K.get_session()
        time.sleep(2)
        
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / self.CLASS, 1., 1.)
                      for x in range(self.CLASS)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None) # Reset seed to default.
    
    
        # Generate output tensor targets for filtered bounding boxes.
        self.yolo_outputs = yolo_head(self.yolo_model.output, self.ANCHORS, self.CLASS)
        self.input_image_shape = K.placeholder(shape=(2, ))
        self.boxes, self.scores, self.classes = yolo_eval(
            self.yolo_outputs,
            self.input_image_shape,
            score_threshold=self.SCORE_THRESHOLD,
            iou_threshold=self.IOU_THRESHOLD)   
        
        self.label_size = 4
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        
        
    def get_single_image(self, image):
     
        in_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        input_image = cv2.resize(in_image, (608, 608))
        
        input_image = input_image/255.
        input_image = input_image[:,:,::-1]
        input_image = np.expand_dims(input_image, 0)
        
    
        
        t0 = time.time()
        
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: input_image,
                self.input_image_shape: [image.shape[0], image.shape[1]],
                K.learning_phase(): 0
            })
       
    
        print('Found {} boxes in {} seconds'.format(len(out_boxes), (time.time()-t0)))
        
        #Make Thicker boxes
        thickness = (image.shape[1] + image.shape[0]) // 300
        
        
        for i, c in reversed(list(enumerate(out_classes))):
            
            predicted_class = self.LABELS[c]
            box = out_boxes[i]
            score = out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)
            print(label, box, c)
            
            if(predicted_class == 'person'):
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

                
                cv2.putText(image, label , (text_origin[0],text_origin[1]), self.font, 0.8, self.colors[c],2,cv2.LINE_AA)
               
                #Make Thicker rectangles
                for i in range(thickness):
                    cv2.rectangle(image, (left + i, top + i ), 
                                  (right - i, bottom - i), 
                                  color = self.colors[c])

        return image, out_boxes, out_scores, out_classes        
        
        
    def start_tracker(self):
        
        #Make the model
        yolo_model = self.get_model()
        tracker.summary()
        
        sess = K.get_session()
        time.sleep(2)
        
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / tracker.CLASS, 1., 1.)
                      for x in range(self.CLASS)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None) # Reset seed to default.
    
    
        # Generate output tensor targets for filtered bounding boxes.
        yolo_outputs = yolo_head(yolo_model.output, self.ANCHORS, self.CLASS)
        input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(
            yolo_outputs,
            input_image_shape,
            score_threshold=self.SCORE_THRESHOLD,
            iou_threshold=self.IOU_THRESHOLD)   
        
        label_size = 4
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        while(True):
            image=cv2.imread('Pictures/jump.jpg')
            in_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            input_image = cv2.resize(in_image, (608, 608))
            
            input_image = input_image/255.
            input_image = input_image[:,:,::-1]
            input_image = np.expand_dims(input_image, 0)
            
        
            
            t0 = time.time()
            
            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    yolo_model.input: input_image,
                    input_image_shape: [image.shape[0], image.shape[1]],
                    K.learning_phase(): 0
                })
            '''
            dummy_array = np.zeros((1,1,1,1,self.TRUE_BOX_BUFFER,4))
            netout = self.model.predict(input_image)
            
            boxes = decode_netout(netout[0], 
                      obj_threshold=self.SCORE_THRESHOLD,
                      nms_threshold=self.IOU_THRESHOLD,
                      anchors=self.ANCHORS, 
                      nb_class=self.CLASS)
            
            image = draw_boxes(image, boxes, labels=self.LABELS)
            '''
            print('Found {} boxes in {} seconds'.format(len(out_boxes), (time.time()-t0)))
            
            #Make Thicker boxes
            thickness = (image.shape[1] + image.shape[0]) // 300
            
            
            for i, c in reversed(list(enumerate(out_classes))):
                
                predicted_class = self.LABELS[c]
                box = out_boxes[i]
                score = out_scores[i]
                label = '{} {:.2f}'.format(predicted_class, score)
                print(label, box, c)
                
                if(predicted_class == 'horse'):
                    top, left, bottom, right = box
                    top = max(0, np.floor(top + 0.5).astype('int32'))
                    left = max(0, np.floor(left + 0.5).astype('int32'))
                    bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
                    right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))
                    
                    
                    #Put label on a smart place
                    if top - label_size >= 0:
                        text_origin = np.array([left, top - label_size])
                    else:
                        text_origin = np.array([left, top + 1])
    
                    
                    cv2.putText(image, label , (text_origin[0],text_origin[1]), font, 0.8, colors[c],2,cv2.LINE_AA)
                   
                    #Make Thicker rectangles
                    for i in range(thickness):
                        cv2.rectangle(image, (left + i, top + i ), 
                                      (right - i, bottom - i), 
                                      color = colors[c])

            
        
        
            cv2.imshow('frame',image)
       
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        

        cv2.destroyAllWindows()

    
        
if __name__ == "__main__":
    
    tracker = Tracker()
    image=cv2.imread('models/research/object_detection/FRAMES/frame3106.png')
    tracker.start_model()
    image, out_boxes, out_scores, out_classes = tracker.get_single_image(image)
    
    cv2.imshow('frame',image)
    
    while(True):
        if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        

   
