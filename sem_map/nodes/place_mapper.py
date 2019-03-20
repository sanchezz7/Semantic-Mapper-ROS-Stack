import os
import numpy as np
import csv
import time
import copy 
import matplotlib.path as mplPath
import threading

from termcolor import colored
import json
from math import pi

import cv2
import rospy
import rospkg
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion
from scene_recognition.msg import DetectedScenes, Scene
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import PointCloud2
from sem_map.msg import Place
from sem_map.srv import places_map_server, places_map_serverResponse
from std_srvs.srv import Trigger, TriggerResponse
import sensor_msgs.point_cloud2 as pc2





Places_cats = ['abbey', 'airport_terminal', 'alley', 'amphitheater', 'amusement_park', 'aquarium', 'aqueduct', 'arch', 'art_gallery', 'art_studio', 'assembly_line', 
						'attic', 'auditorium', 'apartment_building_outdoor', 'badlands', 'ballroom', 'bamboo_forest', 'banquet_hall', 'bar', 'baseball_field', 'basement', 
						'basilica', 'bayou', 'beauty_salon', 'bedroom', 'boardwalk', 'boat_deck', 'bookstore', 'botanical_garden', 'bowling_alley', 'boxing_ring', 'bridge', 
						'building_facade', 'bus_interior', 'butchers_shop', 'butte', 'bakery_shop', 'cafeteria', 'campsite', 'candy_store', 'canyon', 'castle', 'cemetery', 
						'chalet', 'classroom', 'closet', 'clothing_store', 'coast', 'cockpit', 'coffee_shop', 'conference_center', 'conference_room', 'construction_site', 
						'corn_field', 'corridor', 'cottage_garden', 'courthouse', 'courtyard', 'creek', 'crevasse', 'crosswalk', 'cathedral_outdoor', 'church_outdoor', 
						'dam', 'dining_room', 'dock', 'dorm_room', 'driveway', 'desert_sand', 'desert_vegetation', 'dinette_home', 'doorway_outdoor', 'engine_room', 
						'excavation', 'fairway', 'fire_escape', 'fire_station', 'food_court', 'forest_path', 'forest_road', 'formal_garden', 'fountain', 'field_cultivated', 
						'field_wild', 'galley', 'game_room', 'garbage_dump', 'gas_station', 'gift_shop', 'golf_course', 'harbor', 'herb_garden', 'highway', 'home_office', 
						'hospital', 'hospital_room', 'hot_spring', 'hotel_room', 'hotel_outdoor', 'ice_cream_parlor', 'iceberg', 'igloo', 'islet', 'ice_skating_rink_outdoor', 
						'inn_outdoor', 'jail_cell', 'kasbah', 'kindergarden_classroom', 'kitchen', 'kitchenette', 'laundromat', 'lighthouse', 'living_room', 'lobby', 
						'locker_room', 'mansion', 'marsh', 'martial_arts_gym', 'mausoleum', 'medina', 'motel', 'mountain', 'mountain_snowy', 'music_studio', 'market_outdoor',
						'monastery_outdoor', 'museum_indoor', 'nursery', 'ocean', 'office', 'office_building', 'orchard', 'pagoda', 'palace', 'pantry', 'parking_lot', 
						'parlor', 'pasture', 'patio', 'pavilion', 'phone_booth', 'picnic_area', 'playground', 'plaza', 'pond', 'pulpit', 'racecourse', 'raft', 
						'railroad_track', 'rainforest', 'reception', 'residential_neighborhood', 'restaurant', 'restaurant_kitchen', 'restaurant_patio', 'rice_paddy', 'river', 
						'rock_arch', 'rope_bridge', 'ruin', 'runway', 'sandbar', 'schoolhouse', 'sea_cliff', 'shed', 'shoe_shop', 'shopfront', 'shower', 'ski_resort', 
						'ski_slope', 'sky', 'skyscraper', 'slum', 'snowfield', 'staircase', 'supermarket', 'swamp', 'stadium_baseball', 'stadium_football', 'stage_indoor', 
						'subway_station_platform', 'swimming_pool_outdoor', 'television_studio', 'topiary_garden', 'tower', 'train_railway', 'tree_farm', 'trench', 
						'temple_east_asia', 'temple_south_asia', 'track_outdoor', 'train_station_platform', 'underwater_coral_reef', 'valley', 'vegetable_garden', 'veranda', 
						'viaduct', 'volcano', 'waiting_room', 'water_tower', 'watering_hole', 'wheat_field', 'wind_farm', 'windmill', 'yard' ]


class ROOM(object):
	def __init__(self,rid,xmin,xmax,ymin,ymax,belief_method,prior_classes_indexes):
		self.id = rid
		self.xmin = xmin
		self.xmax = xmax
		self.ymin = ymin
		self.ymax = ymax
		self.xcenter = (xmax + xmin)/2
		self.ycenter = (ymax + ymin)/2
		self.belief_method = belief_method
		self.prior_classes_indexes = prior_classes_indexes 
		self.beliefs = [] #list with class id, belief
		self.counter = 0

		if len(self.prior_classes_indexes) > 0:
			self.beliefs = np.ones(205, dtype = np.float32)/205.0 # all the classes start with the same probabilities

		else:
			self.beliefs = np.zeros(205, dtype = np.float32)
			for idx in prior_classes_indexes:
				self.beliefs[idx] = 1.0/len(prior_classes_indexes)

		rospack = rospkg.RosPack()

		
	def update_beliefs(self, new_detection):

		if self.belief_method == 0: #0 iterative mean between all the classes
			for elem in new_detection:
				self.beliefs[elem.id] = ((self.beliefs[elem.id]*self.counter) + elem.probability) / (self.counter + 1)

			self.counter+=1


		elif self.belief_method == 1: #1 iterative mean normalizing network predictions
			detection_values = np.zeros(205, dtype = np.float32) #vector com todas as probabilidades vindas da rede
			for elem in new_detection:
				if elem.id in self.prior_classes_indexes:
					detection_values[elem.id] = elem.probability
			
			detection_values = detection_values / np.linalg.norm(detection_values,ord=1)  #normalize detection values between prior classes
			for idx in self.prior_classes_indexes:
				self.beliefs[idx] = ((self.beliefs[idx]*self.counter) + detection_values[idx]) / (self.counter + 1)

			self.counter+=1


		elif self.belief_method == 2: #right bayesian filtering
			detection_logs = np.zeros(205, dtype = np.float32)
			for elem in new_detection:
				detection_logs[elem.id] = np.log(elem.probability/(1-elem.probability))

			beliefs_logs = np.log(np.divide(self.beliefs,np.subtract(1,self.beliefs))) + detection_logs
			min_log = np.log(0.001/(1-0.001))
			max_log = np.log(0.85/(1-0.85))
			beliefs_logs[beliefs_logs < min_log] = min_log
			beliefs_logs[beliefs_logs > max_log] = max_log

			non_norm_prob = np.divide(np.exp(beliefs_logs), np.add(np.exp(beliefs_logs),1))
			self.beliefs =  np.divide(non_norm_prob, np.sum(non_norm_prob))
			
			self.counter+=1


		elif self.belief_method == 3:
			detection_values = np.zeros(205,dtype = np.float32)
			for elem in new_detection:
				if elem.id in self.prior_classes_indexes:
					detection_values[elem.id] = elem.probability

			detection_values = detection_values / np.sum(detection_values) #normalize cnn outputs between our classes

			detection_values[self.prior_classes_indexes] = np.log( detection_values[self.prior_classes_indexes]/ (1-detection_values[self.prior_classes_indexes]) )

			beliefs_logs = np.zeros(205,dtype = np.float32) 

			beliefs_logs[self.prior_classes_indexes] = np.log( np.divide( self.beliefs[self.prior_classes_indexes], np.subtract(1,self.beliefs[self.prior_classes_indexes] ) ))
			beliefs_logs[self.prior_classes_indexes] += detection_values[self.prior_classes_indexes]

			min_log = np.log(0.01/(1-0.01))
			max_log = np.log(0.925/(1-0.925))
			beliefs_logs[self.prior_classes_indexes and beliefs_logs > max_log] = max_log
			beliefs_logs[self.prior_classes_indexes and beliefs_logs < min_log] = min_log
			non_norm_prob = np.zeros(205, dtype = np.float32)

			non_norm_prob[self.prior_classes_indexes] = np.divide( np.exp(beliefs_logs[self.prior_classes_indexes]), np.add(np.exp( beliefs_logs[self.prior_classes_indexes]),1 ) )

			self.beliefs[self.prior_classes_indexes] = np.divide(non_norm_prob[self.prior_classes_indexes], np.sum(non_norm_prob[self.prior_classes_indexes])) 

			self.counter+=1

		
	def retrieve_ordered_beliefs(self):
		#return sorted(self.beliefs, key = lambda x:x[1], reverse=True)
		return np.argsort(-self.beliefs)

	def retrieve_confs_threshold(self,bef_thresh):
		scene_ids = []
		scene_beliefs = []
		ordered_indexes = np.argsort(-self.beliefs)

		k = 0
		while (self.beliefs[ordered_indexes[k]] >= bef_thresh): #obtain scene categories with probabilities over a certain threshold
			scene_ids.append(ordered_indexes[k])
			scene_beliefs.append(self.beliefs[ordered_indexes[k]])
			k+=1
			if (k==205):
				break

		return scene_ids, scene_beliefs


class SceneMapper(object):

	def __init__(self,sub_pointcloud,sub_detections,pub_markers_topic, categorization_bool, built_scene_map, room_percentage_quality, 
			nan_quality_bool,nan_percentage_quality, segmented_room_file, scenes_representation_file, built_scene_map_continue, belief_method, prior_classes_indexes):

		self.scenes_markers = [] #dictionary with id, label, r,g,b for each class
		with open(scenes_representation_file,'r') as rf:
			csv_reader=csv.DictReader(rf)
			for row in csv_reader:
				self.scenes_markers.append(row)

		self.room_percentage_quality = room_percentage_quality #minimum image's percentage of points layed in a place (only images with high percentage of points layed in a singular place will be used to categorize a specific place)
		self.nan_quality_bool = nan_quality_bool #verify or not nan percentage in image's
		self.nan_percentage_quality = nan_percentage_quality 
		self.segmented_room_file = segmented_room_file #file that describes the map discretization in places

		segmented_rooms = [] #dictionary with the rooms segmented id,xmin,xmax,ymin,ymax,xcenter,ycenter
		with open(segmented_room_file,'r') as rf:
			csv_reader=csv.DictReader(rf)
			for row in csv_reader:
				segmented_rooms.append(row)

		for elem in segmented_rooms:
			elem['id'] = int(float(elem['id']))
			elem['xmin'] = float(elem['xmin'])
			elem['ymin'] = float(elem['ymin'])
			elem['xmax'] = float(elem['xmax'])
			elem['ymax'] = float(elem['ymax'])

		self.room_paths = []
		for room in segmented_rooms:
			room_boundaries = mplPath.Path(np.array([[room['xmin'], room['ymin']],
													[room['xmax'], room['ymin']],
													[room['xmax'], room['ymax']],
													[room['xmin'], room['ymax']]]))
			self.room_paths.append(room_boundaries)

		self.rooms = [] #room class instances with its boundaries, beliefs and so on...
		for j in xrange(0,len(segmented_rooms)): #create room objects -> beliefs are initialized with 0
			if (belief_method == 0) or (belief_method == 2): #no prior classes
				self.rooms.append(ROOM(segmented_rooms[j]['id'],segmented_rooms[j]['xmin'],segmented_rooms[j]['xmax'],segmented_rooms[j]['ymin'],segmented_rooms[j]['ymax'],belief_method, []))
			else: #prior classes (normalize CNN confidences between prior classes)
				self.rooms.append(ROOM(segmented_rooms[j]['id'],segmented_rooms[j]['xmin'],segmented_rooms[j]['xmax'],segmented_rooms[j]['ymin'],segmented_rooms[j]['ymax'],belief_method, prior_classes_indexes))


		self.marker_pub = rospy.Publisher(pub_markers_topic, MarkerArray, queue_size = 1) #publisher of places' markers
		rospy.sleep(0.05)
		self.initialize_markers()
		rospy.sleep(0.05)

		for j in xrange(0,len(segmented_rooms)):
			self.rooms[j].beliefs = np.zeros(205, dtype = np.float32) #initializa numpy array with 0 confidence for each room's scene category 

		if categorization_bool == False: #not categorizing the rooms in an online basis, only loads and uses already known and obtained scene beliefs
			with open(built_scene_map) as scene_file:
				room_scenes = json.load(scene_file) #load pre-obtained scene beliefs 
				
			for j in xrange(0,len(self.rooms)):
				rospy.sleep(0.05) 
				self.update_marker(self.rooms[j].id, room_scenes[j]['scene_id'][0], room_scenes[j]['beliefs'][0]) #update marker with loaded scene beliefs
				for k in xrange(0,len(room_scenes[j]['scene_id'])):
					self.rooms[j].beliefs[room_scenes[j]['scene_id'][k]] = room_scenes[j]['beliefs'][k] #update room instances with the known beliefs
					self.rooms[j].counter = room_scenes[j]['counter'] #update room instances with the known beliefs
					rospy.sleep(0.05)

		else: #categorizing the rooms in an online basis
			self.tfBuffer = tf2_ros.Buffer()
			self.listener = tf2_ros.TransformListener(self.tfBuffer)
			self.locker = threading.Lock()

			if built_scene_map_continue == True: #continue categorizing using previous partial/full categorizations of the environment
				
				with open(built_scene_map) as scene_file:
					room_scenes = json.load(scene_file)

				for j in xrange(0,len(self.rooms)):
					rospy.sleep(0.05)
					self.update_marker(self.rooms[j].id, room_scenes[j]['scene_id'][0], room_scenes[j]['beliefs'][0]) #update marker with loaded scene beliefs
					rospy.sleep(0.05)

					scenes_on_file = []
					belief_counter = 0
					for k in xrange(0,len(room_scenes[j]['scene_id'])):
						scenes_on_file.append(room_scenes[j]['scene_id'][k])
						belief_counter += room_scenes[j]['beliefs'][k]
						self.rooms[j].beliefs[room_scenes[j]['scene_id'][k]] = room_scenes[j]['beliefs'][k] #update room instances
						self.rooms[j].counter = max(room_scenes[j]['counter'],50) #max categorizations used after loading from file = 50
					
					if (belief_method == 1) or (belief_method == 3): 
						idx_not_in_scenes_on_file = [idx for idx in prior_classes_indexes if idx not in scenes_on_file]
						bef_upd = (1-belief_counter)/len(idx_not_in_scenes_on_file)

						for idx in idx_not_in_scenes_on_file:
							self.rooms[j].beliefs[idx] = bef_upd 
					
					else: #categorize between the 205 classes 
						idx_not_in_scenes_on_file = [idx for idx in xrange(0,205) if idx not in scenes_on_file]
						bef_upd = (1-belief_counter)/len(idx_not_in_scenes_on_file)

						for idx in idx_not_in_scenes_on_file:
							self.rooms[j].beliefs[idx] = bef_upd 


			positions_dict = {'r_id':None, 'last_pos':[]}
			self.pos_room_tracker = [] #keeps an history of the locations that were used to classify a place
			
			for j in xrange(0,len(self.rooms)):
				self.pos_room_tracker.append(positions_dict)

			self.pc_xyz_sub = rospy.Subscriber(sub_pointcloud, PointCloud2, self.pointcloud_callback, queue_size=1) #subscribe to pointcloud transformed to the map frame (pc_to_map node created)

			self.detect_sub = rospy.Subscriber(sub_detections, DetectedScenes, self.detections_callback, queue_size=1) #subscribe to the image scene categorizations topic 

			self.scene_map_saver = rospy.Service('~scene_map_saver', Trigger, self.map_saver_response)
				
	
		self.scene_server = rospy.Service('~place_cat_srv', places_map_server, self.server_response)
	

	def server_response(self,request): #replys to a request to obtain the room scene categorie beliefs...
		#response_msg = scenes_server()
		scene_ids = []
		confidences = []
		places = []

		for j in xrange(0,len(self.rooms)):
			place = Place()
			place.xmin = self.rooms[j].xmin
			place.xmax = self.rooms[j].xmax
			place.ymin = self.rooms[j].ymin
			place.ymax = self.rooms[j].ymax
			scene_ids, confidences = self.rooms[j].retrieve_confs_threshold(bef_thresh = request.belief_thresh)
			
			place.scene_ids = np.asarray(scene_ids)
			place.confidences = np.asarray(confidences)

			labels = []
			for k in xrange(0,len(place.scene_ids)): 
				labels.append(copy.deepcopy(self.scenes_markers[place.scene_ids[k]]['label']))

			place.labels = labels
			places.append(copy.deepcopy(place))
				
		return places_map_serverResponse(places = places)


	def map_saver_response(self,request): #saves the current room scene categorie beliefs in a json file
		places_dicts = []

		for j in xrange(0, len(self.rooms)):
			place = {'scene_id': [], 'labels':[], 'beliefs': [], 'xmin': None, 'xmax': None, 'ymin': None, 'ymax': None, 'checked': False, 'counter': 1} 
			place['xmin'] = self.rooms[j].xmin
			place['xmax'] = self.rooms[j].xmax
			place['ymin'] = self.rooms[j].ymin
			place['ymax'] = self.rooms[j].ymax
			place['counter'] = self.rooms[j].counter

			if place['counter'] > 1:
				place['checked'] = True
			else:
				place['checked'] = False


			scene_ids, confidences = self.rooms[j].retrieve_confs_threshold(bef_thresh = 0)
			
			k = 0
			bef = 1.0
			while ((k < 10) and (bef > 0.05)):
				bef = confidences[k]
				if (bef > 0.05):
					place['scene_id'].append(int(scene_ids[k]))
					place['beliefs'].append(float(confidences[k]))
					place['labels'].append(self.scenes_markers[scene_ids[k]]['label'])				
				k+=1

			places_dicts.append(copy.deepcopy(place))

		map_categorized_dir = self.segmented_room_file 
		time_str = time.strftime("%Y,%m,%d,%H,%M,%S").split(',')
		time_str = time_str[2] + '_' +  time_str[1] + '_' + time_str[0] + '|' + time_str[3] + ':' + time_str[4] + ':' + time_str[5]
		file_to_save= map_categorized_dir[:-4] + '|scenes|' + time_str + '.json'

		with open(file_to_save, 'w') as fp:
			json.dump(places_dicts, fp, indent=3)

		return TriggerResponse(success = True, message = 'file saved at {}'.format(file_to_save))


	def pointcloud_callback(self, pointcloud_msg):
		self.locker.acquire()
		self.curr_pc = pointcloud_msg 
		self.locker.release()

	def detections_callback(self,detect_msg):

		bl_pose = PoseStamped() 
		bl_pose.header.frame_id = "base_link" 
		bl_pose.pose.orientation.w = 1.0
		bl_pose.pose.position.x = 0.0
		bl_pose.pose.position.y = 0.0

		transform = self.tfBuffer.lookup_transform("map",
								"base_link", 
								rospy.Time(0), #get the tf at first available time
								rospy.Duration(1.0)) #wait for 1 second

		bot_in_world = tf2_geometry_msgs.do_transform_pose(bl_pose, transform) #get current location of the robot in map frame 
		curr_room_id = self.room_retriever(xpos = float(bot_in_world.pose.position.x), ypos = float(bot_in_world.pose.position.y)) #get current room id

		verify = self.verify_update(bot_in_world = bot_in_world, room_id = curr_room_id, detect_msg = detect_msg) #verify if image has quality to describe a certain place
	

		
	def verify_update(self, bot_in_world, room_id, detect_msg):
		orientation_list = (bot_in_world.pose.orientation.x,bot_in_world.pose.orientation.y,bot_in_world.pose.orientation.z,bot_in_world.pose.orientation.w) #robot orientation in quaternion
		(roll,pitch,yaw) = euler_from_quaternion(orientation_list) #robot orientation in euler angles
		verify = True
		
		for elem in self.pos_room_tracker[room_id]['last_pos']: #if this position was already used, do not use it again
			dist = np.sqrt(((elem['x']-bot_in_world.pose.position.x)**2)+((elem['y']-bot_in_world.pose.position.y)**2))
			if ((abs(elem['yaw'] - yaw ) < (10.0*pi/180)) and (dist < 0.15)): #verify if any position used is nearby the current (15 cm) the new position and same for the angles (10deg)
				verify = False
		
		if verify: #a nearby position was not used, lets verify if this image has a certain quality 
			quality = self.verify_image_quality(current_room = room_id)
			if quality[0] == True: #verifys image quality in terms of its points belonging to a specific place
				new_detection_pose = {'x':bot_in_world.pose.position.x, 'y':bot_in_world.pose.position.y, 'yaw':yaw}
				self.pos_room_tracker[room_id]['last_pos'].append(new_detection_pose) #add new position to the history
				
				if len(self.pos_room_tracker[room_id]['last_pos']) > 50: #keep only 50 locations in history of room updates
					del self.pos_room_tracker[room_id]['last_pos'][0] #delete oldest one

				self.rooms[quality[1]].update_beliefs(detect_msg.detected_scene) #update belief of the observed place (quality[1])
				
				biggest_belief = self.rooms[quality[1]].retrieve_ordered_beliefs() #biggest_belief[0] -> index of the place category with most confidence
				self.update_marker(room_id = quality[1], scene_id = biggest_belief[0], belief = self.rooms[quality[1]].beliefs[biggest_belief[0]]) #update scene's marker
				biggest_belief_prints = self.rooms[quality[1]].retrieve_ordered_beliefs()
				print_arr = [['{}:{}'.format(self.scenes_markers[biggest_belief_prints[k]]['label'],self.rooms[quality[1]].beliefs[biggest_belief_prints[k]])] for k in xrange(2)]
				print print_arr

	
	def verify_image_quality(self, current_room): #verify image quality 
		step_size = 25
		pts=((i,j) for i in xrange(0,self.curr_pc.width,step_size) for j in xrange(0,self.curr_pc.height,step_size)) #pixels of the image to verify in the pointcloud
		
		self.locker.acquire()
		t_ini = time.time()
		points = np.array(list(pc2.read_points(self.curr_pc, field_names = ("x", "y"), skip_nans = True, uvs = pts))) #obtain image's xyz points in the map frame
		self.locker.release()
		#print 'time for acquiring xyz points: {}'.format(time.time()-t_ini)
		
		#do not use all the image points, discretize it and the result should be similiar as place regions are continuous in the image!! this is done to reduce computanial costs
		total_pts = (self.curr_pc.width*self.curr_pc.height)/(step_size**2) #image size divided by step size
		nan_pts = (total_pts - (points.size/2)) #number of nan points
		nan_perc = (float(nan_pts)/total_pts) #nan_percentage of points 
		#print 'total_pts: {}, point.size/2: {}, nan_perc: {}'.format(total_pts,points.size/2,nan_perc)

		if self.nan_quality_bool:
			if nan_perc > self.nan_percentage_quality:
				print colored('nan_perc failed in scene_recog:{:.2f}','red').format(nan_perc)
				return False, 0
				#print 'failed in nan -> nan perc: {}'.format(nan_perc)
				return False, 0 #this image will not be used to classify any place due to the nan values


		
		room_histogram = [0 for elem in self.rooms] #initialize histogram of place's image points
		outrooms = 0 #points that do not belong to any room in the places segmentation file...

		for point in points: #obtain the room's histogram values			
			point_room_id =  self.room_retriever(point[0],point[1]) #obtain point room 
			if point_room_id is None:
				outrooms = outrooms + 1 #one more shitty point
			else:
				room_histogram[point_room_id] += 1 #increment histogram on corresponding place id

		#print 'time for building the place histogram: {}'.format(time.time()-t_init_histogram)
		
		room_histogram = [float(x)/(total_pts-nan_pts-outrooms) for x in room_histogram] #histogram in percentage for the several places
		max_hist = max(room_histogram) 
		maximum_room = (rid for rid, elem in enumerate(room_histogram) if elem >= max_hist)  #obtain most "viewable" place, inspecting the histogram
		maximum_room = list((maximum_room))[0] #mostly observed place id
		
		if room_histogram[maximum_room] > self.room_percentage_quality:
			return True, maximum_room
			#print 'Room_histogram: {}'.format(room_histogram)
			#print colored('Nice image to use to room id:{}!!','green').format(maximum_room)
			#print 'room_perc:{}, nan_perc:{}, outliers_perc:{}'.format(room_histogram[maximum_room]*100, nan_perc, float(outrooms)/total_pts)
			
		else:
			return False, 0
			print colored('Failed in the histogram: {}','red').format(room_histogram)
			

			

	def room_retriever(self, xpos, ypos): #returns place id of a map 2d point
		pt_room = None
		for j in xrange(0,len(self.rooms)):
			if ((xpos < self.rooms[j].xmax) and (xpos > self.rooms[j].xmin) and (ypos < self.rooms[j].ymax) and (ypos > self.rooms[j].ymin)):
				pt_room = j
		return pt_room
		

	def update_marker(self,room_id,scene_id,belief): #function used to update a specif place's marker
		Markers_up = MarkerArray()
		SMarker = Marker()
		Text_Marker = Marker()

		SMarker.header.frame_id = "map"
		SMarker.type = 1
		SMarker.pose.orientation.w = 1.0
		SMarker.lifetime = rospy.Duration(0)
		SMarker.action = SMarker.ADD
		SMarker.pose.position.z = 0
		SMarker.color.r = float(self.scenes_markers[scene_id]['r'])
		SMarker.color.g = float(self.scenes_markers[scene_id]['g'])
		SMarker.color.b = float(self.scenes_markers[scene_id]['b'])
		SMarker.color.a = 0.65
		SMarker.frame_locked = True
		SMarker.id = 1
		SMarker.header.stamp = rospy.Time.now()
		SMarker.ns = '{}-{}'.format(self.rooms[room_id].xcenter, self.rooms[room_id].ycenter)
		SMarker.pose.position.x = self.rooms[room_id].xcenter
		SMarker.pose.position.y = self.rooms[room_id].ycenter
		SMarker.scale.x = self.rooms[room_id].xmax-self.rooms[room_id].xmin
		SMarker.scale.y = self.rooms[room_id].ymax-self.rooms[room_id].ymin
		SMarker.scale.z = 0.1
		Markers_up.markers.append(SMarker)

		Text_Marker.header.frame_id = "map"
		Text_Marker.id = 1
		Text_Marker.type = Text_Marker.TEXT_VIEW_FACING
		Text_Marker.action = 0
		Text_Marker.pose.orientation.w = 1.0
		Text_Marker.scale.z = 0.65
		Text_Marker.color.r = 0
		Text_Marker.color.g = 0
		Text_Marker.color.b = 0
		Text_Marker.color.a = 1
		Text_Marker.lifetime = rospy.Duration(0)
		Text_Marker.frame_locked = True
		Text_Marker.pose.position.z = 0.2

		Text_Marker.header.stamp = rospy.Time.now()
		Text_Marker.ns = 'txt{}-{}'.format(self.rooms[room_id].xcenter,self.rooms[room_id].ycenter)
		Text_Marker.pose.position.x = self.rooms[room_id].xcenter
		Text_Marker.pose.position.y = self.rooms[room_id].ycenter
	
		Text_Marker.text = '{}:{:.2f}%'.format(self.scenes_markers[scene_id]['label'],belief*100)
		#Text_Marker.text = 'Plc:{} ({})'.format(room_id,self.scenes_markers[scene_id]['label'])
	
		Text_Marker.header.stamp = rospy.Time.now()
		Markers_up.markers.append(Text_Marker)
		
		self.marker_pub.publish(Markers_up)

	def initialize_markers(self): #initialize markers for each place at the beginning
		Markersini = MarkerArray()
		for elem in self.rooms:
			SMarker = Marker()
			SMarker.header.frame_id = "map"
			SMarker.type = 1
			SMarker.pose.orientation.w = 1.0
			SMarker.lifetime = rospy.Duration(0)
			SMarker.action = SMarker.ADD
			SMarker.pose.position.z = 0
			SMarker.scale.z = 0.1
			SMarker.color.r = 0.65
			SMarker.color.g = 0.65
			SMarker.color.b = 0.65
			SMarker.color.a = 0.65
			SMarker.frame_locked = True
			SMarker.id = 1
			SMarker.header.stamp = rospy.Time.now()
			SMarker.ns = '{}-{}'.format(elem.xcenter,elem.ycenter)
			SMarker.pose.position.x = elem.xcenter
			SMarker.pose.position.y = elem.ycenter
			SMarker.scale.x = abs(elem.xmax - elem.xmin)
			SMarker.scale.y = abs(elem.ymax - elem.ymin)
			Markersini.markers.append(copy.deepcopy(SMarker))
			print 'inserted room x:{} y:{} marker'.format(elem.xcenter,elem.ycenter)

		for elem in self.rooms:
			Text_Marker = Marker()
			Text_Marker.header.frame_id = "map"
			Text_Marker.id = 1
			Text_Marker.type = Text_Marker.TEXT_VIEW_FACING
			Text_Marker.action = 0
			Text_Marker.pose.orientation.w = 1.0
			Text_Marker.scale.z = 0.45 #/0.65
			Text_Marker.color.r = 0
			Text_Marker.color.g = 0
			Text_Marker.color.b = 0
			Text_Marker.color.a = 1
			Text_Marker.lifetime = rospy.Duration(0)
			Text_Marker.frame_locked = True
			Text_Marker.pose.position.z = 0.2
			Text_Marker.header.stamp = rospy.Time.now()
			Text_Marker.ns = 'txt{}-{}'.format(elem.xcenter,elem.ycenter)
			Text_Marker.pose.position.x = elem.xcenter
			Text_Marker.pose.position.y = elem.ycenter
			Text_Marker.text = '{}: Unknown category!'.format(elem.id)
			Markersini.markers.append(copy.deepcopy(Text_Marker))
			print 'inserted room x:{} y:{} text marker'.format(elem.xcenter,elem.ycenter)

			rospy.sleep(0.1)
		for elem in self.rooms:
			Line_List = Marker()
			Line_List.header.frame_id = "map"
			Line_List.type = 4
			Line_List.pose.orientation.w = 1.0
			Line_List.lifetime = rospy.Duration(0)
			Line_List.action = SMarker.ADD
			Line_List.scale.x = 0.1
			Line_List.color.r = 1
			Line_List.color.g = 0.0
			Line_List.color.b = 0.0
			Line_List.color.a = 0.9
			Line_List.frame_locked = True
			Line_List.id = 1
			Line_List.header.stamp = rospy.Time.now()
			Line_List.ns = 'line{}-{}'.format(elem.xcenter,elem.ycenter)
			
			xmin_ymin = Point()
			xmin_ymin.x = elem.xmin
			xmin_ymin.y = elem.ymin
			xmin_ymin.z = 0.10
			Line_List.points.append(xmin_ymin)

			
			xmax_ymin = Point()
			xmax_ymin.x = elem.xmax
			xmax_ymin.y = elem.ymin
			xmax_ymin.z = 0.10
			Line_List.points.append(xmax_ymin)


			xmax_ymax = Point()
			xmax_ymax.x = elem.xmax
			xmax_ymax.y = elem.ymax
			xmax_ymax.z = 0.10
			Line_List.points.append(xmax_ymax)

			xmin_ymax = Point()
			xmin_ymax.x = elem.xmin
			xmin_ymax.y = elem.ymax
			xmin_ymax.z = 0.10 
			Line_List.points.append(xmin_ymax)
			Line_List.points.append(xmin_ymin)

			Markersini.markers.append(copy.deepcopy(Line_List))
			rospy.sleep(0.1)

		self.marker_pub.publish(Markersini)
		rospy.sleep(0.1)


def main():
	rospy.init_node('place_mapper')
	
	rospack = rospkg.RosPack()

	sub_detections = rospy.get_param('~sub_detections', default = '/scene_recognition_node/detected_scenes')
	pub_markers_topic = rospy.get_param('~pub_markers_topic', default = '~markers')

	segmented_room_file = os.path.join(rospack.get_path('sem_map'),'files',rospy.get_param('~segmented_room_file', default = 'lrm_isr3.csv'))
	scenes_representation_file = os.path.join(rospack.get_path('sem_map'),'files',rospy.get_param('~scenes_represenation_file', default = 'Places205_fullRepresentation.csv'))
	
	room_percentage_quality = rospy.get_param('~room_percentage_quality', default = 0.75)
	nan_quality_bool = rospy.get_param('~nan_quality_bool', default = True)
	nan_percentage_quality = rospy.get_param('~nan_percentage_quality', default = 0.4)

	categorization_bool = rospy.get_param('~categorization_bool', default = True)
	built_scene_map = os.path.join(rospack.get_path('sem_map'),'files',rospy.get_param('~built_scene_map', default='lrm-isr3_categorized.json'))

	built_scene_map_continue = rospy.get_param('~built_scene_map_continue', default = False)

	prior_classes_indexes = rospy.get_param('~prior_classes_indexes', default = [])
	
	belief_method = rospy.get_param('~belief_method', default = 1)

	magician = SceneMapper(sub_pointcloud = 'pcl/map_cloud', sub_detections = sub_detections, pub_markers_topic = pub_markers_topic, categorization_bool = categorization_bool, 
						   room_percentage_quality = room_percentage_quality, nan_quality_bool = nan_quality_bool, nan_percentage_quality = nan_percentage_quality,
						   built_scene_map = built_scene_map, segmented_room_file = segmented_room_file, scenes_representation_file = scenes_representation_file,
						   built_scene_map_continue = built_scene_map_continue, belief_method = belief_method, prior_classes_indexes = prior_classes_indexes)
	

	rospy.spin()
