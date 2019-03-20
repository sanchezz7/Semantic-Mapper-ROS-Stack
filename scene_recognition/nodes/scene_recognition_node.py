import numpy as np
import cv2
import caffe
import os
import csv
import time
import scipy.io 
import rospy
import rospkg
import copy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from scene_recognition.msg import DetectedScenes, Scene




class SceneRecog():
	def __init__(self, imageSubTopic, scenePubTopic, debug, imagePubTopic, model_id, custom_classes, custom_classes_indexes, normalize, CNNMODEL, WEIGHTS, MEAN):

		rospack = rospkg.RosPack()
		#Intialize publishers
		self.scenePubTopic = rospy.Publisher(scenePubTopic,DetectedScenes, queue_size = 1)
		if debug:
			self.imagePubTopic = rospy.Publisher(imagePubTopic,Image, queue_size = 1)

		self.scenes_properties = [] #list of dictionaries with id,label,r,g,b for each scene 
		self.custom_classes = custom_classes

		if self.custom_classes:
			self.custom_classes_indexes = custom_classes_indexes

		with open(os.path.join(rospack.get_path('scene_recognition'),'common','Places205_fullRepresentation.csv'),'r') as rf:
			csv_reader=csv.DictReader(rf)
			for row in csv_reader:
				self.scenes_properties.append(row)

		caffe.set_device(0)
		caffe.set_mode_gpu() 
		self.net = caffe.Classifier(CNNMODEL, WEIGHTS, caffe.TEST)
		if (model_id == 1 or model_id == 2):
			self.transformer = caffe.io.Transformer({'data': (1,3,227,227)})
			self.transformer.set_transpose('data', (2,0,1))
			self.transformer.set_mean('data',MEAN) 
			self.net.blobs['data'].reshape(1,3,227,227)
		else:
			self.transformer = caffe.io.Transformer({'data': (1,3,224,224)})
			self.transformer.set_transpose('data', (2,0,1))
			self.transformer.set_mean('data',MEAN)
			self.net.blobs['data'].reshape(1,3,224,224)
		
		self.debug = debug
		self.model_id = model_id
		self.font = cv2.FONT_HERSHEY_PLAIN
		self.font_size = 2
		self.font_thickness = 5
		self.normalize = normalize
		self.cv_bridge = CvBridge()



		self.image_sub = rospy.Subscriber(imageSubTopic, Image, self.image_callback, queue_size=1, buff_size=2**24)

	def image_callback(self, image_msg):
		#image preprocessing to feed to the network
		
		caffe.set_device(0)
		caffe.set_mode_gpu() 

		cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
		if (self.model_id == 1 or self.model_id ==2):
			cv_image = cv2.resize(cv_image,(227,227))
		else:
			cv_image = cv2.resize(cv_image,(224,224))

		im_input = self.transformer.preprocess('data',cv_image) 
		im_input = im_input[np.newaxis] 
		self.net.blobs['data'].reshape(*im_input.shape)
		self.net.blobs['data'].data[...] = im_input
		
		#get classification results from the desired cnn


		self.net.forward()


		prob  =  self.net.blobs['prob'].data #results in the prob layer
		result = list()
		ids = list()

		if self.custom_classes:
			for j in self.custom_classes_indexes:
				ids.append(j)
				result.append(prob[0,j])
		else: 
			for j in xrange(0,205):
				ids.append(j)
				result.append(prob[0,j])

		if (self.custom_classes and self.normalize): #normalize the beliefs between the desired classes if normalize = true...
			result = np.array(result,np.dtype(float))
			result = result / np.sum(result)

		else: #don't normalize
			result = np.array(result,np.dtype(float))

		results_and_ids = sorted(zip(result,ids), key = lambda x:x[0], reverse=True)



		scenes_detected = DetectedScenes()
		scenes_detected.header.stamp = rospy.Time.now()
		scenes_messages = []
		curr_scene = Scene()

		for elem in results_and_ids:
			curr_scene.id = elem[1]
			curr_scene.label = self.scenes_properties[elem[1]]['label']
			curr_scene.probability = elem[0]
			scenes_detected.detected_scene.append(copy.deepcopy(curr_scene))
		
		self.scenePubTopic.publish(scenes_detected) #send classification results over the ROS network



		if self.debug:
			if self.custom_classes:
				text_x = 10
				text_y = 30
				cv_image = cv2.resize(cv_image,(1280,720))
				for elem in results_and_ids:
					if elem[1] == results_and_ids[0][1]:
						cv2.rectangle(cv_image,(text_x,text_y),(text_x+int(350*float(elem[0])),text_y - 20), (0,255,0),-1)
					else:
						cv2.rectangle(cv_image,(text_x,text_y),(text_x+int(350*float(elem[0])),text_y - 20), (255,0,0),-1)
					cv2.rectangle(cv_image,(text_x,text_y),(text_x+350,text_y - 20), (0,0,0),2)
					text = '{}:{:.1f}%'.format(self.scenes_properties[elem[1]]['label'],float(elem[0]*100))
					cv2.putText(cv_image,text,(text_x,text_y),self.font,1.5,(0,0,0),2)
					text_y += 40
				self.imagePubTopic.publish(self.cv_bridge.cv2_to_imgmsg(cv_image, "bgr8"))

			else:
				text_x = 10
				text_y = 30
				cv_image = cv2.resize(cv_image,(1280,720))
				for j in xrange(0,19):
					if j == 0:
						cv2.rectangle(cv_image,(text_x,text_y),(text_x+int(350*float(results_and_ids[j][0])),text_y -20), (0,255,0),-1)
					else:
						cv2.rectangle(cv_image,(text_x,text_y),(text_x+int(350*float(results_and_ids[j][0])),text_y - 20), (255,0,0),-1)
					cv2.rectangle(cv_image,(text_x,text_y),(text_x+350,text_y - 20), (0,0,0),2)
					text = '{}:{:.1f}%'.format(self.scenes_properties[results_and_ids[j][1]]['label'],float(results_and_ids[j][0]*100))
					cv2.putText(cv_image,text,(text_x,text_y),self.font,1.5,(0,0,0),2)
					text_y += 40
				self.imagePubTopic.publish(self.cv_bridge.cv2_to_imgmsg(cv_image, "bgr8"))

def main():
	rospy.init_node('scene_recog')
	rospack = rospkg.RosPack()

	imageSubTopic = rospy.get_param('~sub_image_topic', default = '/camera/rgb/image_raw')
	scenePubTopic = rospy.get_param('~pub_scenes_topic', default = '~detected_scenes')
	debug = rospy.get_param('~debug', default = True)
	imagePubTopic = rospy.get_param('~pub_image_topic', default = '~scenes_images')
	model_id = rospy.get_param('~model_id', default = 3)   
	custom_classes = rospy.get_param('~custom_classes', default = True)
	custom_classes_indexes = rospy.get_param('~custom_classes_indexes', default = [11,18,24,44,54,64,70,93,108,112,113,129,134,135,136,138,139,167,174,195,198])
	normalize = rospy.get_param('~normalize', default = True)

	if (model_id == 1):
		CNNMODEL = os.path.join(rospack.get_path('scene_recognition'),'models','GoogleNet','deploy_places205.prototxt')
		WEIGHTS = os.path.join(rospack.get_path('scene_recognition'),'models','GoogleNet','googlelet_places205_train_iter_2400000.caffemodel')
		MEAN = np.array([104, 117, 123])

	elif (model_id == 2):
		CNNMODEL = os.path.join(rospack.get_path('scene_recognition'),'models','AlexNet','places205CNN_deploy.prototxt')
		WEIGHTS = os.path.join(rospack.get_path('scene_recognition'),'models','AlexNet','places205CNN_iter_300000.caffemodel')
		MEAN = scipy.io.loadmat(os.path.join(rospack.get_path('scene_recognition'),'models','AlexNet','places_mean.mat'))
		MEAN = np.mean(np.array(MEAN['image_mean']), axis =tuple(range(0,2)))

	elif (model_id == 3):
		CNNMODEL = os.path.join(rospack.get_path('scene_recognition'),'models','VGG16','siat_scene_vgg_16_deploy.prototxt')
		WEIGHTS = os.path.join(rospack.get_path('scene_recognition'),'models','VGG16','siat_scene_vgg_16.caffemodel')
		MEAN = scipy.io.loadmat(os.path.join(rospack.get_path('scene_recognition'),'models','VGG16','places205_mean.mat'))
		MEAN = np.mean(np.array(MEAN['image_mean']), axis =tuple(range(0,2)))

	else:
		CNNMODEL = os.path.join(rospack.get_path('scene_recognition'),'models','VGG19','siat_scene_vgg_19_deploy.prototxt')
		WEIGHTS = os.path.join(rospack.get_path('scene_recognition'),'models','VGG19','siat_scene_vgg_19.caffemodel')
		MEAN = scipy.io.loadmat(os.path.join(rospack.get_path('scene_recognition'),'models','VGG19','places205_mean.mat'))
		MEAN = np.mean(np.array(MEAN['image_mean']), axis =tuple(range(0,2)))

	magician = SceneRecog(imageSubTopic = imageSubTopic, scenePubTopic = scenePubTopic, debug = debug, imagePubTopic = imagePubTopic, model_id = model_id, custom_classes = custom_classes,
						custom_classes_indexes = custom_classes_indexes, normalize = normalize, CNNMODEL = CNNMODEL, WEIGHTS = WEIGHTS, MEAN = MEAN)
	rospy.spin()
