from ctypes import *
import math
import random
import cv2
import csv
import copy
import time
import rospy
import os
import time
from object_detection.msg import Object, Detections
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as IM
import rospkg
import copy

#code retired from darknet examples -> dog.jpg classification example, reused and integrated in ros
#you need to change your libdarknet.so path 


def sample(probs):
	s = sum(probs)
	probs = [a/s for a in probs]
	r = random.uniform(0, 1)
	for i in range(len(probs)):
		r = r - probs[i]
		if r <= 0:
			return i
	return len(probs)-1

def c_array(ctype, values):
	arr = (ctype*len(values))()
	arr[:] = values
	return arr

class BOX(Structure):
	_fields_ = [("x", c_float),
				("y", c_float),
				("w", c_float),
				("h", c_float)]

class DETECTION(Structure):
	_fields_ = [("bbox", BOX),
				("classes", c_int),
				("prob", POINTER(c_float)),
				("mask", POINTER(c_float)),
				("objectness", c_float),
				("sort_class", c_int)]


class IMAGE(Structure):
	_fields_ = [("w", c_int),
				("h", c_int),
				("c", c_int),
				("data", POINTER(c_float))]

class METADATA(Structure):
	_fields_ = [("classes", c_int),
				("names", POINTER(c_char_p))]

def classify(net, meta, im):
	out = predict_image(net, im)
	res = []
	for i in range(meta.classes):
		res.append((meta.names[i], out[i]))
	res = sorted(res, key=lambda x: -x[1])
	return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
	im = load_image(image, 0, 0)
	num = c_int(0)
	pnum = pointer(num)
	predict_image(net, im)
	dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
	num = pnum[0]
	if (nms): do_nms_obj(dets, num, meta.classes, nms);

	res = []
	for j in range(num):
		for i in range(meta.classes):
			if dets[j].prob[i] > 0:
				b = dets[j].bbox
				res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
	res = sorted(res, key=lambda x: -x[1])
	free_image(im)
	free_detections(dets, num)
	return res



#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/home/some1/darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

class DarknetRecog(object):
	def __init__(self, imageSubTopic, objPubTopic, debug, BbInfoFile, imagePubTopic, detectThresh, cfgFile, weightsFile, dataFile, tmp_image):

		self._img_path = tmp_image
		self._objPubTopic = rospy.Publisher(objPubTopic,Detections, queue_size=1)
		self._detectThresh = detectThresh
		self._debug = debug
		if debug == True:
			self._imagePubTopic = rospy.Publisher(imagePubTopic,IM, queue_size=1)
		self._cv_bridge = CvBridge() 

		self._ObjectsDescriptors = []
		with open(BbInfoFile,'r') as rf:
			csv_reader = csv.DictReader(rf)
			for row in csv_reader:
				self._ObjectsDescriptors.append(row)

		self._font = cv2.FONT_HERSHEY_PLAIN
		self._lineThickness = 4



		self._net = load_net(cfgFile, weightsFile, 0)
		self._meta = load_meta(dataFile)
		self._sub = rospy.Subscriber(imageSubTopic, IM, self.callback, queue_size=1, buff_size=2**24)

		self._labels = []
		for j in xrange(0,self._meta.classes):
			self._labels.append(self._meta.names[j])

	def callback(self, image_msg):
		t_ini = time.time()
		cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
		cv2.imwrite(self._img_path,cv_image)
		res = detect(self._net, self._meta, self._img_path)
		os.remove(self._img_path)
		

		objects_detected = Detections()
		objects_detected.header.stamp = rospy.Time.now()
		objects_messages = []
		curr_object = Object()

		boxes = []
		detection = {'label':None, 'confidence':None, 'xmin':None, 'xmax':None, 'ymin':None, 'ymax':None}
		for elem in res:
			if elem[1] > self._detectThresh: #and (self._labels.index(elem[0]) not in [72,60]))
				
				if self._labels.index(elem[0]) not in [72,60,13,15,25,27,26,71,79]:

					detection['label'] = elem[0]
					detection['confidence'] = elem[1]
					detection['xmin'] = int(elem[2][0] - (elem[2][2]/2))
					detection['xmax'] = int(elem[2][0] + (elem[2][2]/2))
					detection['ymin'] = int(elem[2][1] - (elem[2][3]/2))
					detection['ymax'] = int(elem[2][1] + (elem[2][3]/2))

					if (detection['xmin'] < 0):
						detection['xmin'] = 0

					if (detection['xmax'] > 640):
						detection['xmax'] = 640

					if (detection['ymin'] < 0):
						detection['ymin'] = 0

					if (detection['ymax'] > 480):
						detection['ymax'] = 480

					boxes.append(copy.deepcopy(detection))
					#print 'Label: {} p:{} xmin:{} xmax:{} ymin:{} ymax{}'.format(detection['label'],detection['confidence'],detection['xmin'],detection['xmax'],detection['ymin'],detection['ymax'])
					curr_object.id = self._labels.index(detection['label'])
					curr_object.label = detection['label']
					curr_object.probability = detection['confidence']
					curr_object.center_x = int((detection['xmax'] + detection['xmin'])/2)
					curr_object.center_y = int((detection['ymax'] + detection['ymin'])/2)
					curr_object.width_min = int(detection['xmin'])  #xmin 
					curr_object.height_max = int(detection['ymax']) #ymax -> bottom y
					curr_object.width_max = int(detection['xmax']) #xmax
					curr_object.height_min = int(detection['ymin'])  #ymi
					objects_detected.objects.append(copy.deepcopy(curr_object))

		self._objPubTopic.publish(objects_detected)

		if self._debug == True:
			for elem in boxes:
				marker_index = self._labels.index(elem['label'])
				cv2.line(cv_image, (elem['xmin'], elem['ymin']), (elem['xmax'], elem['ymin']), (int(float(self._ObjectsDescriptors[marker_index]['r'])*255),
					int(float(self._ObjectsDescriptors[marker_index]['g'])*255),int(float(self._ObjectsDescriptors[marker_index]['b'])*255)), self._lineThickness)
				cv2.line(cv_image, (elem['xmin'], elem['ymax']), (elem['xmax'], elem['ymax']), (int(float(self._ObjectsDescriptors[marker_index]['r'])*255),
					int(float(self._ObjectsDescriptors[marker_index]['g'])*255),int(float(self._ObjectsDescriptors[marker_index]['b'])*255)), self._lineThickness)
				cv2.line(cv_image, (elem['xmin'], elem['ymin']), (elem['xmin'], elem['ymax']), (int(float(self._ObjectsDescriptors[marker_index]['r'])*255),
					int(float(self._ObjectsDescriptors[marker_index]['g'])*255),int(float(self._ObjectsDescriptors[marker_index]['b'])*255)), self._lineThickness)
				text = '{}:{:.1f}%'.format(elem['label'],float(elem['confidence']*100))
				cv2.line(cv_image, (elem['xmax'], elem['ymin']), (elem['xmax'], elem['ymax']), (int(float(self._ObjectsDescriptors[marker_index]['r'])*255),
					int(float(self._ObjectsDescriptors[marker_index]['g'])*255),int(float(self._ObjectsDescriptors[marker_index]['b'])*255)), self._lineThickness)
				cv2.rectangle(cv_image,(elem['xmin'],elem['ymin']-1),(elem['xmin']+int(len(text)*10.5),elem['ymin']+15),
					  (int(float(self._ObjectsDescriptors[marker_index]['r'])*255),
					   int(float(self._ObjectsDescriptors[marker_index]['g'])*255),int(float(self._ObjectsDescriptors[marker_index]['b'])*255)),-1)
				
				cv2.putText(cv_image,text,(elem['xmin'],elem['ymin']+15),self._font,1.1,(0,0,0),1)
			
			self._imagePubTopic.publish(self._cv_bridge.cv2_to_imgmsg(cv_image, "bgr8"))

		


def main():

	rospy.init_node('object_recog_darknet')
	rospack = rospkg.RosPack()

	imageSubTopic = rospy.get_param('~sub_image_topic', default = '/camera/rgb/image_raw')
	objPubTopic = rospy.get_param('~pub_objects_topic', default = '~detected_objects')
	debug = rospy.get_param('~debug', default = True)
	imagePubTopic = rospy.get_param('~pub_image_topic', default = '~image_detections')
	detectThresh = rospy.get_param('~detections_thresh', default = 0.5)
	cfgFile = os.path.join(rospack.get_path('object_detection'),'models','darknet','cfg',rospy.get_param('~cfg_file', default = 'yolov3608.cfg'))
	weightsFile = os.path.join(rospack.get_path('object_detection'),'models','darknet','weights',rospy.get_param('~weights_file', default = 'yolov3608.weights'))
	dataFile =  os.path.join(rospack.get_path('object_detection'),'models','darknet','data',rospy.get_param('~data_file', default = 'coco.data'))
	BbInfoFile = os.path.join(rospack.get_path('object_detection'),'models','darknet','bb_file',rospy.get_param('~bb_file', default ='objects_data.csv'))

	tmp_image = os.path.join(rospack.get_path('object_detection'),'detect.jpg')
	recog = DarknetRecog(imageSubTopic = imageSubTopic, objPubTopic = objPubTopic, debug = debug, BbInfoFile = BbInfoFile, imagePubTopic = imagePubTopic, detectThresh = detectThresh,
		cfgFile = cfgFile, weightsFile = weightsFile, dataFile = dataFile, tmp_image = tmp_image) 

	rospy.spin()
