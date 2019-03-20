#!/usr/bin/env python
# coding=utf-8
import rospy
from roslib import message
import numpy as np
import csv
import os
import time
from visualization_msgs.msg import Marker
from object_detection.msg import Object, Detections
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped
import tf2_ros
import tf2_geometry_msgs
import sensor_msgs.point_cloud2 as pc2
import math
from scipy.stats import mode
import time
import rospkg
import threading
import copy
from termcolor import colored
import json
from std_srvs.srv import Trigger, TriggerResponse
from sem_map.msg import ObjectInMap
from sem_map.srv import ObjectInstances, ObjectInstancesResponse


class ObjectOccurances(object):
	def __init__(self,object_type,label,position,belief,MarkerView,MarkerText,publisher,markerheight):
		self.object_type = object_type #id of the object
		self.label = label #label
		self.position = position #dictionary with x,y,z of object in the world
		self.belief = belief #belief of its occurance
		self.detections = 1 #number of occurances detected
		self.MarkerView = MarkerView #PShape Marker
		self.MarkerText = MarkerText #Text Marker
		self.marker_pub = publisher #Topic to publish Marker 
		self.markerheight = markerheight
		self.publish_new_object_marker() #publish new marker function
		 

	def update_belief(self,detect_conf,seen):
		#C_(k) = min (  ( detection confidence * inc_ratio + C_(k-1) ), max_confidence )
		max_confidence = 3.0 #limit object occurrence confidence to this value
		inc_ratio = 0.8 #increase ratio
		dec_ratio = 0.45 #decrease ratio
		if seen == True: 
			aux = (self.belief + (inc_ratio*detect_conf))
			self.belief = min(aux, max_confidence)
		else:
			self.belief -= dec_ratio
		
	
	def publish_new_object_marker(self):
		self.marker_pub.publish(self.MarkerView)
		self.marker_pub.publish(self.MarkerText)
		print 'published new marker'

	def update_position(self, new_x_pos, new_y_pos, new_z_pos):
		#iterative mean

		self.position['x'] = ((self.position['x']*self.detections)+new_x_pos)/(self.detections+1)
		self.position['y'] = ((self.position['y']*self.detections)+new_y_pos)/(self.detections+1)
		self.position['z'] = ((self.position['z']*self.detections)+new_z_pos)/(self.detections+1)
		self.detections += 1

		self.update_marker()

	def update_marker(self):
		self.MarkerText.pose.position.x = self.position['x']
		self.MarkerText.pose.position.y = self.position['y']
		self.MarkerText.pose.position.z = self.position['z']+self.markerheight

		self.MarkerView.pose.position.x = self.position['x']
		self.MarkerView.pose.position.y = self.position['y']
		self.MarkerView.pose.position.z = self.position['z']

		self.marker_pub.publish(self.MarkerView)
		self.marker_pub.publish(self.MarkerText)

	def delete_markers(self):
		self.MarkerView.action = self.MarkerView.DELETE
		self.marker_pub.publish(self.MarkerView)
		self.MarkerText.action = self.MarkerText.DELETE
		self.marker_pub.publish(self.MarkerText)


class ObjectMapper(object):
	def __init__(self, pub_markers_topic, sub_detections_topic, sub_pointcloud_topic, ObjMarkersFile, resolution, 
				 object_map_loader, object_map_file, registration):

		self.ObjectsDescriptors = [] #dictionary with id,label,r,g,b,a,x_scale,y_scale,z_scale, MarkerType helpfull variables to work with the markers
		with open(ObjMarkersFile,'r') as rf:
			csv_reader=csv.DictReader(rf)
			for row in csv_reader:
				self.ObjectsDescriptors.append(row)

		for elem in self.ObjectsDescriptors: #load object marker characteristics for each object category
			elem['r']=float(elem['r'])
			elem['g']=float(elem['g'])
			elem['b']=float(elem['b'])
			elem['a']=float(elem['a'])
			elem['x_scale']=float(elem['x_scale'])
			elem['y_scale']=float(elem['y_scale'])
			elem['z_scale']=float(elem['z_scale'])
			elem['MarkerType']=int(float(elem['MarkerType']))
			elem['id']=int(float(elem['id']))

		self.marker_pub = rospy.Publisher(pub_markers_topic, Marker, queue_size = 200)
		self.Objects_In_World = [] #list of ObjectOcurrances elements
		self.Objects_In_World_sem = threading.Lock() #semaphore initialization to deal with simultaneous accessing and writing in Objects_In_World

		if object_map_loader == True: #load objects map already constructed from object_map_file
			with open(object_map_file) as objects_file:
				objects_in_map = json.load(objects_file)

			rospy.sleep(0.5)
			for idx, obj in enumerate(objects_in_map):
				rospy.sleep(0.001)
				self.add_new_object_to_world(object_id = obj['id'], localization = obj['pos'], belief = obj['confidence'])
				self.Objects_In_World[idx].detections = obj['ndetections']


		if registration == True:
			self.min_depth_add = 0.65 #minimum depth distance to add object to world
			self.min_depth_delete = 1.0 #minimum depth distance to decrease an object's occurrence confidence
			self.max_depth_add = 4.0  #maximum depth distance to add object to world
			self.max_depth_delete = 3.0 #maximum depth distance decrease an object's occurrence confidence
			self.good_pts_perc = 0.65 #minimum percentage of non nan pts in bounding box to register the object from that detection in the map  
			self.max_distance = resolution #max distance to say its the same object as seen before		
			self.pc_xyz_sub = rospy.Subscriber(sub_pointcloud_topic, PointCloud2, self.pointcloud_callback, queue_size=1)
			self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0)) #tf buffer length
			self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
			self.detect_sub = rospy.Subscriber(sub_detections_topic, Detections, self.od_detections_callback, queue_size=1) 

		self.obj_map_saver = rospy.Service('~/obj_map_saver', Trigger, self.map_saver_server)
		self.obj_instances_server = rospy.Service('~/obj_instances', ObjectInstances, self.obj_server)



	def map_saver_server(self,request): #saves the current known objects in a json file
		self.Objects_In_World_sem.acquire()
		objects_dicts = [] #list of objects to be saved

		for j in xrange(0, len(self.Objects_In_World)):
			obj = {'id': None, 'label': None, 'pos': None, 'confidence': None, 'ndetections': None} #an object to be saved

			obj['id'] =  self.Objects_In_World[j].object_type
			obj['label'] = self.Objects_In_World[j].label
			obj['pos'] = {'x':self.Objects_In_World[j].position['x'], 'y':self.Objects_In_World[j].position['y'], 'z':self.Objects_In_World[j].position['z']}
			obj['confidence'] = self.Objects_In_World[j].belief
			obj['ndetections'] = self.Objects_In_World[j].detections

			objects_dicts.append(copy.deepcopy(obj))

		self.Objects_In_World_sem.release()

		time_str = time.strftime("%Y,%m,%d,%H,%M,%S").split(',')
		time_str = time_str[2] + '_' +  time_str[1] + '_' + time_str[0] + '|' + time_str[3] + ':' + time_str[4] + ':' + time_str[5]
		file_to_save =  'objects_map' + time_str + '.json'

		with open(file_to_save, 'w') as fp:
			json.dump(objects_dicts, fp, indent=3)

		return TriggerResponse(success = True, message = "objects' map file saved at {}".format(file_to_save))


	def obj_server(self, request):
		objects_to_retrieve = []
		obj_instance = ObjectInMap()
		self.Objects_In_World_sem.acquire()
		
		if request.req_id == 1: #search for specific objects
			for obj in self.Objects_In_World:
				if obj.object_type in request.object_id:
					obj_instance.id = obj.object_type
					obj_instance.pos_x = obj.position['x']
					obj_instance.pos_y = obj.position['y']
					obj_instance.pos_z = obj.position['z']
					objects_to_retrieve.append(copy.deepcopy(obj_instance))

		
		elif request.req_id == 2: #search objects in a circular radius between the robot (z coordinate is not taken into account)
			bot_on_bl = PoseStamped()
			bot_on_bl.header.frame_id = 'base_link'
			bot_on_bl.pose.orientation.w = 1.0
			bot_on_bl.pose.position.pos_x = 0.0
			bot_on_bl.pose.position.pos_y = 0.0
			bot_on_bl.pose.position.pos_z = 0.0
			
			transform = self.tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0))
			bot_on_map = tf2_geometry_msgs.do_transform_pose(bot_on_bl, transform) #point on camera_depth_optical_frame

			for obj in self.Objects_In_World: 
				dist = np.sqrt( ((bot_on_map.pose.position.x - obj.position['x'])**2) + ((bot_on_map.pose.position.y - obj.position['y'])**2) )
				if dist <= request.radius:
					obj_instance.id = obj.object_type
					obj_instance.pos_x = obj.position['x']
					obj_instance.pos_y = obj.position['y']
					obj_instance.pos_z = obj.position['z']
					objects_to_retrieve.append(copy.deepcopy(obj_instance))


		elif request.req_id == 3: #search objects in a specified bounded area
			for obj in self.Objects_In_World:
				if ( (obj.position['x'] > request.xmin) and (obj.position['x'] < request.xmax) and (obj.position['y'] > request.ymin) and (obj.position['y'] < request.ymax) ):
					obj_instance.id = obj.object_type
					obj_instance.pos_x = obj.position['x']
					obj_instance.pos_y = obj.position['y']
					obj_instance.pos_z = obj.position['z']
					objects_to_retrieve.append(copy.deepcopy(obj_instance))

		else: # request id set to 0 ornone of the above... send full object list 
			for obj in self.Objects_In_World:
				obj_instance.id = obj.object_type
				obj_instance.pos_x = obj.position['x']
				obj_instance.pos_y = obj.position['y']
				obj_instance.pos_z = obj.position['z']
				objects_to_retrieve.append(copy.deepcopy(obj_instance))


		self.Objects_In_World_sem.release()

		return ObjectInstancesResponse (objects_in_map = objects_to_retrieve)



	def pointcloud_callback(self, pointcloud_msg):
		self.curr_pc = pointcloud_msg


	def od_detections_callback(self, detections_msg):
		
		self.Objects_In_World_sem.acquire()
		used_indx = []
		print colored('New Detection CB!!!!!!!','green')

		detect_occ_pair = [None] * len(detections_msg.objects) 
		detect_occ_dist = [None] * len(detections_msg.objects)
		occ_detect_pair = [None] * len(self.Objects_In_World)
		occ_detect_distance = [None] * len(self.Objects_In_World)
		n_obj_init_occurrences = len(self.Objects_In_World)
		
		detections_reaction = [0] * len(detections_msg.objects)
		objs_localizations = []

		for j in xrange(0,len(detections_msg.objects)): #iterate over all objects detected and check if they are already represented
			detection = detections_msg.objects[j]
			objs_localizations.append(copy.deepcopy(self.get_object_3D_point_bb(width_min=detection.width_min, width_max=detection.width_max, height_min=detection.height_min, height_max=detection.height_max, histogram_method='fd')))
			
			if len(self.Objects_In_World)> 0:
				if objs_localizations[j]['points_verf'] == True:
					for k in xrange(0,len(self.Objects_In_World)): #confront detections with objects already registered
						object_world = self.Objects_In_World[k]
						if detection.id == object_world.object_type:
							distance = np.sqrt(((objs_localizations[j]['x_pos'] - object_world.position['x'])**2)+((objs_localizations[j]['y_pos']-object_world.position['y'])**2)+((objs_localizations[j]['z_pos']-object_world.position['z'])**2))
							if (distance < self.max_distance): #distance to confront detection with an object already registered on the semantic map
								print 'detect_location:{}'.format([objs_localizations[j]['x_pos'],objs_localizations[j]['y_pos'],objs_localizations[j]['z_pos']])
								print 'obj_occur:{}'.format([object_world.position['x'],object_world.position['y'],object_world.position['z']])
								print 'distance:{}'.format(distance) 
									
								if detect_occ_pair[j] != None:
									if distance < detect_occ_dist[j]:
										#print 'print got an occurence more near'
										detect_occ_pair[j] = k #founded an object occurence that is near than the other one
										detect_occ_dist[j] = distance
										if occ_detect_pair[k] != None:
											if occ_detect_distance[k] > distance: #'got a detection in a nearest neighborhood'
												detections_reaction[occ_detect_pair[k]] = 2 
											
												occ_detect_pair[k] = j
												occ_detect_distance[k] = distance												
										else:
											occ_detect_pair[k] = j
											occ_detect_distance[k] = distance
												
								else:
									detections_reaction[j] = 1
									detect_occ_pair[j] = k
									detect_occ_dist[j] = distance
									if occ_detect_pair[k] != None:
										if occ_detect_distance[k] > distance: #got a detection in a nearest neighborhood'
											detections_reaction[occ_detect_pair[k]] = 2 
											occ_detect_pair[k] = j
											occ_detect_distance[k] = distance												
									else:
										occ_detect_pair[k] = j
										occ_detect_distance[k] = distance
				else:
					detect_occ_pair[j]=2


		n_added_objects = 0
		print 'reacts:{}'.format(detections_reaction)							
		for j in xrange(0,len(detections_msg.objects)):
			if detections_reaction[j] == 1:  #update object occurences confronted beliefs
				self.Objects_In_World[int(detect_occ_pair[j])].update_belief(detect_conf = detections_msg.objects[j].probability, seen = True)	#confidence update
				self.Objects_In_World[int(detect_occ_pair[j])].update_position(new_x_pos = objs_localizations[j]['x_pos'], new_y_pos = objs_localizations[j]['y_pos'], new_z_pos = objs_localizations[j]['z_pos']) #position update
				
			if detections_reaction[j] == 0: #add objects occurences, need to verify if in the buffer there's not another detection of the same class nearby with higher confidence
				adding_flag = True
				print 'nan_perc:{}'.format(objs_localizations[j]['nan_perc'])
				if ( (objs_localizations[j]['cam_depth'] < (self.min_depth_add)) or (objs_localizations[j]['cam_depth'] > (self.max_depth_add)) or (objs_localizations[j]['nan_perc'] > self.good_pts_perc) ) :
					adding_flag = False #object detection location is not safe to be registered on the semantic map
					print 'set add flag false, cam_depth:{}, r_o_dist:{}, nan_perc:{}'.format(objs_localizations[j]['cam_depth'],objs_localizations[j]['r_o_dist'],objs_localizations[j]['nan_perc'])

				if adding_flag == True:
					localization={'x':objs_localizations[j]['x_pos'],'y':objs_localizations[j]['y_pos'],'z':objs_localizations[j]['z_pos']}
					self.add_new_object_to_world(object_id = int(float(detections_msg.objects[j].id)), localization = localization, belief = float(detections_msg.objects[j].probability))
					print 'adding object'
					n_added_objects += 1

		
		#Decrease confidences of objects not in the FOV and not occluded
		if (n_obj_init_occurrences > 0):
			objects_to_delete_indexes = []
			transform = self.tf_buffer.lookup_transform('camera_depth_optical_frame', 'map', rospy.Time(0))
			for j in xrange(0,len(self.Objects_In_World) - n_added_objects):
				if occ_detect_pair[j] == None:
					if self.verify_field_of_view_and_occlusion_deleting(self.Objects_In_World[j].position['x'],self.Objects_In_World[j].position['y'],self.Objects_In_World[j].position['z'],transform = transform):
						self.Objects_In_World[j].update_belief(detect_conf = 0, seen = False) #decrease belief of this object
						#print colored('Object Not Occluded and not seen:','red') + "{}".format(object_world.label)
						if self.Objects_In_World[j].belief <= 0: 
							objects_to_delete_indexes.append(j)
					
			objects_to_delete_indexes.sort(reverse=True) #sort indexes descending in order to easily delete wanted objects
			for del_index in objects_to_delete_indexes:
				self.Objects_In_World[del_index].delete_markers()
				del self.Objects_In_World[del_index] #delete desired objects

		self.Objects_In_World_sem.release()


	def verify_field_of_view_and_occlusion_deleting(self,object_x_pos,object_y_pos,object_z_pos,transform):

		object_on_world = PoseStamped()
		object_on_world.header.frame_id = 'map'
		object_on_world.pose.orientation.w = 1.0
		object_on_world.pose.position.x = object_x_pos
		object_on_world.pose.position.y = object_y_pos
		object_on_world.pose.position.z = object_z_pos

		#transform = self.tf_buffer.lookup_transform('camera_depth_optical_frame', 'map', rospy.Time(0))
		object_on_camera = tf2_geometry_msgs.do_transform_pose(object_on_world, transform) #point on camera_depth_optical_frame	
		if object_on_camera.pose.position.z < self.min_depth_delete or object_on_camera.pose.position.z > self.max_depth_delete:
			return False #object not in the acceptable depth values for being detected, therefore we will not delete it
		

		fovx = 52.0 #fovx depth sensor
		fovy = 40.0 #fovy depth sensor
		projection_matrix = np.matrix('525 0 319.5 0; 0 525 239.5 0; 0 0 1 0') #intrinsic parameters from depth camera
		xangle = np.arctan(object_on_camera.pose.position.x/object_on_camera.pose.position.z) #rads
		yangle = np.arctan(object_on_camera.pose.position.y/object_on_camera.pose.position.z) #rads		
		if ( ( abs(xangle) > abs(fovx/2*math.pi/180) ) or (abs(yangle) > abs(fovy/2*math.pi/180) ) ):
			return False #object not in the camera's angles, dont delete it

		
		camera_pose = np.array([object_on_camera.pose.position.x, object_on_camera.pose.position.y, object_on_camera.pose.position.z, 1])
		uvw = projection_matrix.dot(camera_pose) #obtain projected pixel from object's 3D localization

		x_pixel = uvw[0,0]/uvw[0,2]
		y_pixel = uvw[0,1]/uvw[0,2]
		
		window_size = 10 #generate window of pixels to verify depth values
		x_major = x_pixel+window_size
		x_minor = x_pixel-window_size
		y_major = y_pixel+window_size
		y_minor = y_pixel-window_size

		if x_major > self.curr_pc.width:
			x_major = self.curr_pc.width-1
		if x_minor < 0:
			x_minor = 0

		if y_major > self.curr_pc.height:
			y_major = self.curr_pc.height-1
		if y_minor < 0:
			y_minor = 0
			
		bb_pts = ((i,j) for i in xrange(int(x_minor),int(x_major)) for j in xrange(int(y_minor), int(y_major)))
		depth_values = np.array(list(pc2.read_points(self.curr_pc, field_names = ("z"), skip_nans=True, uvs=bb_pts)))

		if depth_values.size == 0: #got it all nans in the projected pixels for the object
			return False #object should be occluded

		occ_thresh = 0.075
		if np.any(depth_values < (object_on_camera.pose.position.z - occ_thresh)): 
			return False #object occluded
			#print colored('Occlusion Verification:','red') + 'x_pixel:{}, y_pixel:{}'.format(x_pixel,y_pixel)
			#print colored ('Object is occluded','red')
		else:
			return True #object is not occluded

	
	def get_object_3D_point_bb(self,width_min,width_max,height_min,height_max,histogram_method):
		step_size = 3
		bb_pts=((i,j) for i in xrange(width_min,width_max,step_size) for j in xrange(height_min, height_max,step_size))#pixels of the image to verify in the pointcloud
		points = np.array(list(pc2.read_points(self.curr_pc, field_names = ("x", "y", "z"), skip_nans=True, uvs=bb_pts)))

		rect_dict = {'x_pos':None, 'y_pos':None, 'z_pos':None, 'nan_perc':None, 'xangle':None, 'yangle':None, 'cam_depth':None, 'r_o_dist':None, 'points_verf':True}
		rect_dict['nan_perc'] = (1.0 - float(float(points.size/3)/float((width_max-width_min)*(height_max-height_min)/(step_size**2))))

		if (points.size!=0): #sometimes it gets 0 points... ALL NANS?!?
			z_min = min(points[:,2]) #minimum distance to the objected acepted to annotate the object
			z_max = max(points[:,2]) #max depth value observed

			if histogram_method == 'None':
				hist_step = 0.25 #histogram "step" (0.25m)
				nbins = np.ceil((z_max - z_min)/hist_step) #number of histogram classes
				bins = [(i*hist_step)+z_min for i in xrange(0,int(nbins)+1)] #histogram classes
				z_hist, edges = np.histogram(points[:,2], bins, range=(z_min,z_max)) #histogram of depth values

			else:
				z_hist, bins = np.histogram(points[:,2], bins=histogram_method, range=(z_min,z_max)) #histogram of depth values

			z_points_bins = np.digitize(points[:,2],bins) #the indice of the bin for each point (0 is the first interval)
			z_mode, countss = mode(z_points_bins)
			points_indexes = [index for index, point_bin in enumerate(z_points_bins) if point_bin == z_mode]
			cam_position = np.mean(points[points_indexes,:],axis=0) #get the mean of x,y,z's -> final object localization estimation prediction for this object detection  (in the camera's depth optical frame)
			x_pos = cam_position[0]
			y_pos = cam_position[1]
			z_pos = cam_position[2]
			rect_dict['cam_depth'] = z_pos

			#fill the posestamped with the position variables, so that we can transform them to world later on 	
			depth_optical_frame_pose = PoseStamped() #pose stamped of the object point in the camera_depth_optical frame 
			depth_optical_frame_pose.header.frame_id = "camera_depth_optical_frame"
			depth_optical_frame_pose.pose.orientation.w = 1.0
	
			depth_optical_frame_pose.header.stamp = rospy.Time.now() 
			depth_optical_frame_pose.pose.position.x = x_pos
			depth_optical_frame_pose.pose.position.y = y_pos
			depth_optical_frame_pose.pose.position.z = z_pos


			transform = self.tf_buffer.lookup_transform("map",
							"camera_depth_optical_frame", 
							rospy.Time(0)) 

			pose_transformed = tf2_geometry_msgs.do_transform_pose(depth_optical_frame_pose, transform) #final localization estimation in the map frame

			rect_dict['x_pos'] = pose_transformed.pose.position.x
			rect_dict['y_pos'] = pose_transformed.pose.position.y
			rect_dict['z_pos'] = pose_transformed.pose.position.z
			rect_dict['r_o_dist'] = np.sqrt((depth_optical_frame_pose.pose.position.x**2)+(depth_optical_frame_pose.pose.position.y**2)+(depth_optical_frame_pose.pose.position.z**2))
			rect_dict['yangle'] = np.arctan(depth_optical_frame_pose.pose.position.y/depth_optical_frame_pose.pose.position.z) #rads
			rect_dict['xangle'] = np.arctan(depth_optical_frame_pose.pose.position.x/depth_optical_frame_pose.pose.position.z) #rads

			return rect_dict
		else:
			rect_dict = {'x_pos':None, 'y_pos':None, 'z_pos':None, 'nan_perc':None, 'xangle':None, 'yangle':None, 'cam_depth':None, 'r_o_dist':None, 'points_verf':False}
			return rect_dict

	def add_new_object_to_world(self,object_id,localization,belief):

		OMarker = Marker()
		OMarker.header.stamp = rospy.Time.now()
		OMarker.header.frame_id = "map"
		OMarker.ns = '{}-{}-{}'.format(localization['x'],localization['y'],localization['z'])
		OMarker.id = object_id
		OMarker.type = self.ObjectsDescriptors[object_id]['MarkerType']
		OMarker.action = OMarker.ADD #0 add/modify an object, 1 (deprecated), 2 deletes an object, 3 deletes all objects
		OMarker.pose.position.z = localization['z']

		#don't allow objects to go bellow z = 0, this is need cause of poor localizations and because of random scale used for objects
		if ((localization['z'] - (self.ObjectsDescriptors[object_id]['z_scale']/2)) < 0):
			OMarker.pose.position.z = self.ObjectsDescriptors[object_id]['z_scale']/2

		OMarker.pose.position.x = localization['x']
		OMarker.pose.position.y = localization['y']
		#OMarker.pose.position.z = mindfucker
		OMarker.pose.orientation.w = 1.0
		OMarker.scale.x = self.ObjectsDescriptors[object_id]['x_scale']
		OMarker.scale.y = self.ObjectsDescriptors[object_id]['y_scale']
		OMarker.scale.z = self.ObjectsDescriptors[object_id]['z_scale']
		OMarker.color.r = self.ObjectsDescriptors[object_id]['r']
		OMarker.color.g = self.ObjectsDescriptors[object_id]['g']
		OMarker.color.b = self.ObjectsDescriptors[object_id]['b']
		OMarker.color.a = 0.90 #self.ObjectsDescriptors[object_id]['a']
		OMarker.lifetime = rospy.Duration(0)
		OMarker.frame_locked = 1

		Text_Marker = Marker()
		Text_Marker.header.stamp = rospy.Time.now()
		Text_Marker.header.frame_id = "map"
		Text_Marker.ns = 'txt{}-{}-{}'.format(localization['x'],localization['y'],localization['z'])
		Text_Marker.id = 1
		Text_Marker.type = Text_Marker.TEXT_VIEW_FACING
		Text_Marker.action = Text_Marker.ADD
		Text_Marker.pose.position.x = localization['x']
		Text_Marker.pose.position.y = localization['y']
		Text_Marker.pose.position.z = localization['z']+((self.ObjectsDescriptors[object_id]['z_scale'])/2) #put text in top of the box
		Text_Marker.pose.orientation.w = 1.0
		Text_Marker.scale.z = 0.20
		Text_Marker.color.r = 0
		Text_Marker.color.g = 0
		Text_Marker.color.b = 0
		Text_Marker.color.a = 1
		Text_Marker.lifetime = rospy.Duration(0)
		Text_Marker.frame_locked = 1
		Text_Marker.text = self.ObjectsDescriptors[object_id]['label']
		
		new_object = ObjectOccurances(object_type = object_id, label = self.ObjectsDescriptors[object_id]['label'], position = localization, belief = belief, 
			MarkerView = OMarker , MarkerText = Text_Marker , publisher = self.marker_pub, markerheight = (self.ObjectsDescriptors[object_id]['z_scale']/2))

		self.Objects_In_World.append(new_object)


def main():	
	rospy.init_node('object_mapper_node')
	rospack = rospkg.RosPack()

	pub_markers_topic = rospy.get_param('~pub_markers_topic', default = '~obj_markers')
	sub_detections_topic = rospy.get_param('~sub_detections', default = '/object_recognition_darknet_node/detected_objects')
	sub_pointcloud_topic = rospy.get_param('~sub_pointcloud', default = '/camera/depth_registered/points')
	object_map_loader = rospy.get_param('~object_map_loader', default = False)
	object_map_file = os.path.join(rospack.get_path('sem_map'),'files',rospy.get_param('~object_map_file', default = 'lrm_isr|objects|22_10_2018|22:57:50.json'))

	resolution = rospy.get_param('~resolution', default = 0.7)
	registration = rospy.get_param('~registration', default = True)
	ObjMarkersFile = os.path.join(rospack.get_path('sem_map'),'files', rospy.get_param('~markers_file', default ='objects_data.csv'))

	obj_mapper = ObjectMapper(pub_markers_topic = pub_markers_topic, sub_detections_topic = sub_detections_topic, sub_pointcloud_topic = sub_pointcloud_topic, 
		ObjMarkersFile = ObjMarkersFile, resolution = resolution, object_map_loader = object_map_loader, object_map_file = object_map_file, registration = registration)

	rospy.spin()
