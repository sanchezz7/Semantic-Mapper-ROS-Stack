# coding=utf-8
import rospy
import numpy as np
import os
import sys
import tensorflow as tf
import cv2 
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from object_detection.msg import Object, Detections
import time
import rospkg
from copy import deepcopy
import od_ops as utils_ops
import od_label_map_util
import od_visualization_utils as vis_util



class ObjectRecognition(object):
	
	def __init__(self, imageSubTopic, objPubTopic, debug, imagePubTopic, detectThresh, visualizeThresh, labelsFile, modelFile):
	
 
		self._objPubTopic = rospy.Publisher(objPubTopic,Detections, queue_size=1)
		self._detectThresh = detectThresh
		self._debug = debug
		if debug == True:
			self._imagePubTopic = rospy.Publisher(imagePubTopic,Image, queue_size=1) #if debug announce topic to publish the image with detections
		self._cv_bridge = CvBridge()
		self._visualizeThresh = visualizeThresh

		self._detection_graph = tf.Graph() #begin the tensorflow graph
		with self._detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(modelFile, 'rb') as fid: #load the pre-trained model
				serialized_graph = fid.read()  
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')

		label_map = od_label_map_util.load_labelmap(labelsFile) #labels to all the coco classes
		categories = od_label_map_util.convert_label_map_to_categories(label_map, max_num_classes = 90, use_display_name=True) #label_map to categories
	
		self._category_index = od_label_map_util.create_category_index(categories) #indexes of the categories


		with self._detection_graph.as_default():
			self._sess = tf.Session(graph=self._detection_graph) #create the tf session 
			# Define input and output Tensors for detection_graph
			self._image_tensor = self._detection_graph.get_tensor_by_name('image_tensor:0')
			# Each box represents a part of the image where a particular object was detected.
			self._detection_boxes = self._detection_graph.get_tensor_by_name('detection_boxes:0')
			# Each score represent how level of confidence for each of the objects.
			# Score is shown on the result image, together with the class label.
			self._detection_scores = self._detection_graph.get_tensor_by_name('detection_scores:0')
			self._detection_classes = self._detection_graph.get_tensor_by_name('detection_classes:0')
			self._num_detections = self._detection_graph.get_tensor_by_name('num_detections:0')


		self.ids_correction = {'1':0, '2':1, '3':2, '4':3, '5':4, '6':5, '7':6,
							   '8':7, '9':8, '10':9, '11':10, '13':11, '14':12,
							   '15':13, '16':14, '17':15, '18':16, '19':17, '20':18,
							   '21':19, '22':20, '23':21, '24':22, '25':23, '27':24,
							   '28':25, '31':26, '32':27, '33':28, '34':29, '35':30,
							   '36':31, '37':32, '38':33, '39':34, '40':35, '41':36,
							   '42':37, '43':38, '44':39, '46':40, '47':41, '48':42,
							   '49':43, '50':44, '51':45, '52':46, '53':47, '54':48,
							   '55':49, '56':50, '57':51, '58':52, '59':53, '60':54,
							   '61':55, '62':56, '63':57, '64':58, '65':59, '67':60,
							   '70':61, '72':62, '73':63, '74':64, '75':65, '76':66,
							   '77':67, '78':68, '79':69, '80':70, '81':71, '82':72,
							   '84':73, '85':74, '86':75, '87':76, '88':77, '89':78,
							   '90':79}


		self._sub = rospy.Subscriber(imageSubTopic, Image, self.callback, queue_size=1, buff_size=2**24) #subscribe image topic


	def callback(self, image_msg):
		
		cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
		image_np_expanded = np.expand_dims(cv_image, axis=0) 

		(boxes, scores, classes, num) = self._sess.run(
			[self._detection_boxes, self._detection_scores, self._detection_classes, self._num_detections], feed_dict = {self._image_tensor: image_np_expanded}) #get the results

		if self._debug == True:
			vis_util.visualize_boxes_and_labels_on_image_array(cv_image, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), 
				self._category_index, min_score_thresh = self._visualizeThresh, use_normalized_coordinates = True, line_thickness = 8)
			self._imagePubTopic.publish(self._cv_bridge.cv2_to_imgmsg(cv_image, "bgr8"))
		
		
		#process the data given by the network
		classes = np.squeeze(classes).astype(np.int32)
		scores = np.squeeze(scores)
		boxes = np.squeeze(boxes)

		objects_detected = Detections() #create the message to be published
		objects_messages = [] #array with the objects detections -> needed for detections message
		curr_object = Object()

		i = 0
		top_score = scores[0]
		while top_score > self._detectThresh:
			class_name = self._category_index[classes[i]]['name']
			#print 'lab:{}, id:{}, p:{}, y_min:{}, x_min:{}, y_max:{}, x_max:{}, y_c:{}, x_c:{}'.format(class_name, classes[i], scores[i], boxes[i][0]*image_msg.height, boxes[i][1]*image_msg.width, boxes[i][2]*image_msg.height, boxes[i][3]*image_msg.width, (boxes[i][2]+boxes[i][0])/2*image_msg.height, (boxes[i][3]+boxes[i][1])/2*image_msg.width)
			curr_object.id = self.ids_correction[str(classes[i])]
			curr_object.label = class_name
			curr_object.probability = scores[i]
			#print int((boxes[i][3]+boxes[i][1])/2*image_msg.width)
			curr_object.center_x = int((boxes[i][3]+boxes[i][1])/2*image_msg.width)
			curr_object.center_y = int((boxes[i][2]+boxes[i][0])/2*image_msg.height)
			curr_object.width_min = int(boxes[i][1]*image_msg.width)  #xmin 
			curr_object.width_max = int(boxes[i][3]*image_msg.width) #xmax
			curr_object.height_min = int(boxes[i][0]*image_msg.height)  #ymin -> top y
			curr_object.height_max = int(boxes[i][2]*image_msg.height) #ymax -> bottom y				
			objects_detected.objects.append(deepcopy(curr_object))
			i+=1
			top_score = scores[i]
		

		objects_detected.header.stamp = rospy.Time.now()
		self._objPubTopic.publish(objects_detected)


def main():

	rospy.init_node('object_recog_tensorflow')
	rospack = rospkg.RosPack()

	imageSubTopic = rospy.get_param('~sub_image_topic', default = '/camera/rgb/image_raw') #image topic to subscribe
	objPubTopic = rospy.get_param('~pub_objects_topic', default = '~detected_objects') #object detections topic to publish
	debug = rospy.get_param('~debug', default = False) #if debug set to true you can visualize the objects being recognised in imagepubtopic
	imagePubTopic = rospy.get_param('~pub_image_topic', default = '~image_detections')
	detectThresh = rospy.get_param('~detections_thresh', default = 0.5) #only objects with belief bigger than this will be published
	visualizeThresh = rospy.get_param('~visualize_thresh', default = 0.5) #used in debug
	labelsFile = os.path.join(rospack.get_path('object_detection'),'models','tensorflow','dataset_labels',rospy.get_param('~labels_file', default = 'mscoco_label_map.pbtxt')) #classes/labels for cnn
	modelFile = os.path.join(rospack.get_path('object_detection'),'models','tensorflow','trained_models',rospy.get_param('~model_file', default = 'ssd_inception_v2_coco_2017_11_17.pb')) #cnn pretrained model 

	recognise = ObjectRecognition(imageSubTopic = imageSubTopic, objPubTopic = objPubTopic, debug = debug, imagePubTopic = imagePubTopic,
		detectThresh = detectThresh, visualizeThresh = visualizeThresh, labelsFile = labelsFile, modelFile = modelFile) #let the magic happens ;)
  
	rospy.spin()

