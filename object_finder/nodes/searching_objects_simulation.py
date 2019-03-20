import rospy
import rospkg
from object_recognition.msg import Detections
from sem_map.msg import ObjectInMap, Place
from sem_map.srv import scenes_server, scenes_serverResponse
from std_srvs.srv import Trigger, TriggerResponse
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
import sensor_msgs.point_cloud2 as pc2
from scipy.stats import mode
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import cv2
import actionlib

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseActionResult, MoveBaseActionFeedback

import numpy as np 
import matplotlib.pyplot as plt
import os
from termcolor import colored
import time
from copy import deepcopy

class ObjectSearcher(object):
	
	def __init__(self, obj_label, obj_poses, reasoning_algorithm, plc_in_map_srv, obj_finder_srv_namespace, pub_markers_topic, 
				 ObjConcepts, PlacesConcepts, SuperObjectConcepts, SuperObjObj, ObjPlace, ObjObj):
		
		self._ObjConcepts = ObjConcepts #list of object labels
		self._PlacesConcepts = PlacesConcepts #list of places labels
		self._SuperObjectConcepts = SuperObjectConcepts #list of superconcepts labels [vehicles, animals, food, electronics]
		self._SuperObjObj = SuperObjObj #super objects to objects relations (e.g. apple is a food)
										#boolean np.array dim(4,80) (superconcept, object) describing if there is relation or not between the object and the super category
		self._ObjPlace = ObjPlace # boolean np.array describing the relations between objects and places (80,205) (AtLocationRelation)
		self._ObjObj = ObjObj #boolean numpy array describing NearbyRelation (80,80) apple Nearby Banana = True, therefore self._ObjObj[apple_idx, banana_idx] = True
		
		self._cv_bridge = CvBridge()
		self.tfBuffer = tf2_ros.Buffer()
		self.listener = tf2_ros.TransformListener(self.tfBuffer)
		
		self.obj_label = obj_label 
		self.obj_poses = obj_poses

		self.min_goal_distance = 0.30
		self.state_machine = 0
		#state 0 -> waiting for searching request 
		#state 1 -> moving to a destination to search
		#state 2 -> rotating in a desired position 

		self.stop_counter = 0
		self.init_rotation_timer = False
		self.travelling_timer = 0

		if reasoning_algorithm == True:
			self.obj_finder_server = rospy.Service('~reasoning_search', Trigger, self.object_search_reasoning)

		else:
			self.obj_finder_server = rospy.Service('~bruteforce_search', Trigger, self.object_search_bruteforce)

		
		self.mb_status = rospy.Subscriber('move_base/status', actionlib.GoalStatusArray, self.mb_status_cb, queue_size = 1)

		self.cancel_pub = rospy.Publisher('move_base/cancel', actionlib.GoalID, queue_size = 10)

		self.velocity_publisher = rospy.Publisher('/cmd_vel_mux/input/navi', Twist, queue_size = 1)
		self.marker_pub = rospy.Publisher(pub_markers_topic, Marker, queue_size = 200)

		self.move_base_client = actionlib.SimpleActionClient('move_base',MoveBaseAction)
		print 'waiting for move_base server'
		self.move_base_client.wait_for_server()
		print 'move_base available'

		rospy.wait_for_service(plc_in_map_srv)
		try:
			self.plc_in_map_srv = rospy.ServiceProxy(plc_in_map_srv, scenes_server)
		except rospy.ServiceException, e:
			print "Service call failed: %s"%e

		print 'scene mapper server availble'


	def marker_handler(self,action,x,y): #type 0 for navigation searching goals, 1 for found object, action 0 for adding, action 1 for removing
		Mk = Marker()
		Mk.header.frame_id = "map"
		Mk.id = 1
		Mk.type = Mk.SPHERE

		txt_marker = Marker()
		txt_marker.header.frame_id = "map"
		txt_marker.id = 1
		txt_marker.type = txt_marker.TEXT_VIEW_FACING
		
		obj_radius = 1.25
		obj_pose_z = 0.75
		goal_pose_z = 1.25
		goal_radius = 0.75

		if action == 0:
			for pose in self.obj_poses:
				Mk.action = Mk.ADD
				Mk.pose.position.x = pose[0]
				Mk.pose.position.y = pose[1]
				Mk.pose.position.z = obj_pose_z
				Mk.scale.x = obj_radius
				Mk.scale.y = obj_radius
				Mk.scale.z = obj_radius
				Mk.color.r = 1.0
				Mk.color.g = 1.0
				Mk.color.b = 0.0
				Mk.color.a = 0.8
				Mk.lifetime = rospy.Duration(0)
				Mk.ns = 'obj_{:.1f}:{:.1f}'.format(pose[0],pose[1])
				Mk.pose.orientation.w = 1.0
				Mk.frame_locked = 1
				Mk.header.stamp = rospy.Time.now()
				self.marker_pub.publish(Mk)
				rospy.sleep(0.005)

				txt_marker.action = txt_marker.ADD
				txt_marker.pose.position.x = pose[0] 
				txt_marker.pose.position.y = pose[1]
				txt_marker.pose.position.z = obj_pose_z + obj_radius
				txt_marker.ns = 'obj_txt_{:.1f}:{:.1f}'.format(pose[0],pose[1]) #pose increment radius
				txt_marker.text = self.obj_label
				txt_marker.color.r = 0.0
				txt_marker.color.g = 0.0
				txt_marker.color.b = 0.0
				txt_marker.lifetime = rospy.Duration(0)
			 	txt_marker.color.a = 1
				txt_marker.pose.orientation.w = 1.0
				txt_marker.frame_locked = 1
				txt_marker.scale.z = 0.75
				txt_marker.color.a = 1
				txt_marker.header.stamp = rospy.Time.now()
				self.marker_pub.publish(txt_marker)
				rospy.sleep(0.005)


		elif action == 2:
			for j in xrange(0,len(self.rooms_to_search)):
				Mk.action = Mk.ADD
				Mk.pose.position.x = self.rooms_to_search[j]['xcenter']
				Mk.pose.position.y = self.rooms_to_search[j]['ycenter']
				Mk.pose.position.z = goal_pose_z
				Mk.scale.x = goal_radius
				Mk.scale.y = goal_radius
				Mk.scale.z = goal_radius
				
				if j == 0:
					Mk.color.r = 1.0
					Mk.color.g = 0.4975
					Mk.color.b = 0.0
					Mk.color.a = 1.0
				else:
					Mk.color.r = 0.0
					Mk.color.g = 0.0
					Mk.color.b = 1.0
					Mk.color.a = 0.8


				Mk.lifetime = rospy.Duration(0)
				Mk.ns = 'plc_{:.1f}:{:.1f}'.format(self.rooms_to_search[j]['xcenter'],self.rooms_to_search[j]['ycenter'])
				Mk.pose.orientation.w = 1.0
				Mk.frame_locked = 1
				Mk.header.stamp = rospy.Time.now()
				self.marker_pub.publish(Mk)
				rospy.sleep(0.005)

				txt_marker.action = txt_marker.ADD
				txt_marker.pose.position.x = self.rooms_to_search[j]['xcenter'] 
				txt_marker.pose.position.y = self.rooms_to_search[j]['ycenter']
				txt_marker.pose.position.z = goal_pose_z + goal_radius
				txt_marker.ns = 'plc_txt_{:.1f}:{:.1f}'.format(self.rooms_to_search[j]['xcenter'],self.rooms_to_search[j]['ycenter']) #pose increment radius
				if j == 0:
					txt_marker.text = 'Current Searching Goal'
					txt_marker.scale.z = 0.75
				else:
					txt_marker.text = 'Searching Goal'
					txt_marker.scale.z = 0.75

				txt_marker.color.r = 0.0
				txt_marker.color.g = 0.0
				txt_marker.color.b = 0.0
				txt_marker.lifetime = rospy.Duration(0)
			 	txt_marker.color.a = 1
				txt_marker.pose.orientation.w = 1.0
				txt_marker.frame_locked = 1

				txt_marker.color.a = 1
				txt_marker.header.stamp = rospy.Time.now()
				self.marker_pub.publish(txt_marker)
				rospy.sleep(0.005)
	
		
		elif action == 3:
			Mk.action = Mk.DELETE
			Mk.ns = 'plc_{:.1f}:{:.1f}'.format(x,y) 
			Mk.header.stamp = rospy.Time.now()
			self.marker_pub.publish(Mk)
			rospy.sleep(0.005)

			txt_marker.action = txt_marker.DELETE 
			txt_marker.ns = 'plc_txt_{:.1f}:{:.1f}'.format(x,y)
			txt_marker.header.stamp = rospy.Time.now()
			self.marker_pub.publish(txt_marker)
			rospy.sleep(0.005)

		elif action == 4:

			Mk.action = Mk.ADD
			Mk.pose.position.x = x
			Mk.pose.position.y = y
			Mk.pose.position.z = goal_pose_z
			Mk.color.r = 1.0
			Mk.color.g = 0.4975
			Mk.color.b = 0.0
			Mk.color.a = 1.0

			Mk.scale.x = goal_radius
			Mk.scale.y = goal_radius
			Mk.scale.z = goal_radius
			Mk.lifetime = rospy.Duration(0)
			Mk.ns = 'plc_{:.1f}:{:.1f}'.format(x,y)
			Mk.pose.orientation.w = 1.0
			Mk.frame_locked = 1
			Mk.header.stamp = rospy.Time.now()
			self.marker_pub.publish(Mk)
			rospy.sleep(0.005)

			txt_marker.action = txt_marker.ADD
			txt_marker.pose.position.x = x
			txt_marker.pose.position.y = y
			txt_marker.pose.position.z = Mk.pose.position.z + goal_radius
			txt_marker.ns = 'plc_txt_{:.1f}:{:.1f}'.format(x,y) #pose increment radius
			txt_marker.text = 'Current Searching Goal'
			txt_marker.color.r = 0.0
			txt_marker.color.g = 0.0
			txt_marker.color.b = 0.0
			txt_marker.color.a = 1
			txt_marker.scale.z = 0.75
			txt_marker.lifetime = rospy.Duration(0)
			txt_marker.pose.orientation.w = 1.0
			txt_marker.frame_locked = 1
			
			txt_marker.header.stamp = rospy.Time.now()
			self.marker_pub.publish(txt_marker)
			rospy.sleep(0.005)

		elif action == 1:
			
			for pose in self.obj_poses:
				if ((x!=pose[0]) or (y!=pose[1])):
					Mk.action = Mk.DELETE
					Mk.ns = 'obj_{:.1f}:{:.1f}'.format(pose[0],pose[1]) 
					Mk.header.stamp = rospy.Time.now()
					self.marker_pub.publish(Mk)
					rospy.sleep(0.005)

					txt_marker.action = Mk.DELETE
					txt_marker.ns = 'obj_txt_{:.1f}:{:.1f}'.format(pose[0],pose[1]) 
					txt_marker.header.stamp = rospy.Time.now()
					self.marker_pub.publish(txt_marker)
					rospy.sleep(0.005)
				
				else:
					Mk.action = Mk.ADD
					Mk.pose.position.x = x
					Mk.pose.position.y = y
					Mk.pose.position.z = obj_pose_z
					Mk.color.r = 0.0
					Mk.color.g = 1.0
					Mk.color.b = 0.0
					Mk.color.a = 1.0
					Mk.scale.x = obj_radius + 0.15
					Mk.scale.y = obj_radius + 0.15
					Mk.scale.z = obj_radius + 0.15
					Mk.lifetime = rospy.Duration(30)
					Mk.ns = 'obj_{:.1f}:{:.1f}'.format(x,y)
					Mk.pose.orientation.w = 1.0
					Mk.frame_locked = 1
					Mk.header.stamp = rospy.Time.now()
					self.marker_pub.publish(Mk)
					rospy.sleep(0.005)

					txt_marker.action = txt_marker.ADD
					txt_marker.pose.position.x = x 
					txt_marker.pose.position.y = y
					txt_marker.pose.position.z = obj_pose_z + obj_radius
					txt_marker.ns = 'obj_txt_{:.1f}:{:.1f}'.format(x,y) #pose increment radius
					txt_marker.text = 'Founded {}'.format(self.obj_label)
					txt_marker.color.r = 0.0
					txt_marker.color.g = 0.0
					txt_marker.color.b = 0.0
					txt_marker.lifetime = rospy.Duration(30)
					txt_marker.color.a = 1
					txt_marker.pose.orientation.w = 1.0
					txt_marker.frame_locked = 1
					txt_marker.scale.z = 0.75
					txt_marker.header.stamp = rospy.Time.now()
					self.marker_pub.publish(txt_marker)
					rospy.sleep(0.005)


			for scene in self.rooms_to_search:

				Mk.action = Mk.DELETE
				Mk.lifetime = rospy.Duration(0)
				Mk.ns = 'plc_{:.1f}:{:.1f}'.format(scene['xcenter'],scene['ycenter'])
				Mk.header.stamp = rospy.Time.now()
				self.marker_pub.publish(Mk)
				rospy.sleep(0.005)
				
				txt_marker.action = txt_marker.DELETE
				txt_marker.ns = 'plc_txt_{:.1f}:{:.1f}'.format(scene['xcenter'],scene['ycenter']) #pose increment radius
				txt_marker.lifetime = rospy.Duration(0)
				txt_marker.header.stamp = rospy.Time.now()
				self.marker_pub.publish(txt_marker)
				rospy.sleep(0.005)

	



	def object_search_reasoning(self, request):

		places_related = set( np.argwhere( self._ObjPlace[self._ObjConcepts.index(self.obj_label), :] ).ravel())
		if len(places_related) > 0:
			print colored(self.obj_label,'cyan'), ' can be ', colored('AtLocation: ','blue'), colored([self._PlacesConcepts[place] for place in places_related],'green')

		objects_related_from_nearby = np.argwhere( self._ObjObj[self._ObjConcepts.index(self.obj_label), :] ).ravel()
		plcs = []
		
		if len(objects_related_from_nearby) > 0:
			print colored(self.obj_label, 'cyan'), ' can be found ', colored('NearBy: ','blue'), colored([self._ObjConcepts[obj] for obj in objects_related_from_nearby], 'green')
			for obj_related in objects_related_from_nearby:
				nb_places_related = np.argwhere(self._ObjPlace[obj_related, :]).ravel()
					
				if len(nb_places_related) > 0:
					print colored(self._ObjConcepts[obj_related],'magenta'), colored(' AtLocation: ','blue'), colored([self._PlacesConcepts[plc] for plc in nb_places_related],'green')
					for elem in nb_places_related:
						plcs.append(elem)
				
			if len(plcs) > 0:
				places_set = set()
				places_add = places_set.add
				plcs = [x for x in plcs if not (x in places_set or places_add(x))]
				print 'Therefore ', colored(self.obj_label,'cyan'), 'might also be ', colored('AtLocation: ','blue'), colored([self._PlacesConcepts[plc] for plc in plcs], 'green')

					
		if ( (len(objects_related_from_nearby) == 0) and (len(places_related) == 0) ):
			print colored( 'I do not where such object could be located', 'red')

		else:
			bl_pose = PoseStamped() 
			bl_pose.header.frame_id = "base_link" 
			bl_pose.pose.orientation.w = 1.0
			bl_pose.pose.position.x = 0.0
			bl_pose.pose.position.y = 0.0

			transform = self.tfBuffer.lookup_transform("map", "base_link", rospy.Time(0), rospy.Duration(1.0))
			
			bot_in_world = tf2_geometry_msgs.do_transform_pose(bl_pose, transform) #get current location of the robot in map frame 
			
			self.rooms_to_search = []
			
			response = self.plc_in_map_srv(0.90)
			room_dict = {'xmin':None, 'xmax':None, 'ymin':None, 'ymax':None, 'xcenter':None, 'ycenter':None, 'distance':None, 'scene_ids':[], 'confidences':[]}

			for room in response.places: 
			
				direct_identique = places_related & set(room.scene_ids)
				nearby_identique = set(plcs) & set(room.scene_ids)
				room_classes_to_search = direct_identique | nearby_identique 

				if len(room_classes_to_search) > 0:
					room_dict['xmin'] = room.xmin
					room_dict['xmax'] = room.xmax
					room_dict['ymin'] = room.ymin
					room_dict['ymax'] = room.ymax
					room_dict['xcenter'] = (room.xmax + room.xmin)/2
					room_dict['ycenter'] = (room.ymax + room.ymin)/2
					room_dict['distance'] = np.sqrt( np.power(bot_in_world.pose.position.x - room_dict['xcenter'],2) + np.power(bot_in_world.pose.position.y - room_dict['ycenter'], 2) )
					
					for scene_id in room_classes_to_search:
						if scene_id not in room_dict['scene_ids']:
							idx_in_room = room.scene_ids.index(scene_id)
							room_dict['scene_ids'].append(scene_id)
							room_dict['confidences'].append(room.confidences[idx_in_room])
					self.rooms_to_search.append(deepcopy(room_dict))


			self.rooms_to_search = sorted(self.rooms_to_search, key=lambda k:k['distance'])
			self.marker_handler(action = 0, x = None, y = None)
			self.marker_handler(action = 2, x = None, y = None)

			print 'init_goals:{}'.format(self.rooms_to_search)

			print colored('Moving to x:{:.2f}, y{:.2f}','green').format(self.rooms_to_search[0]['xcenter'],self.rooms_to_search[0]['ycenter'])
			self.travelling_timer = time.time()
			goal = MoveBaseGoal()
			goal.target_pose.header.frame_id = "map"
			goal.target_pose.header.stamp = rospy.Time.now()
			goal.target_pose.pose.position.x = self.rooms_to_search[0]['xcenter']
			goal.target_pose.pose.position.y = self.rooms_to_search[0]['ycenter']
			goal.target_pose.pose.orientation.w = 1.0
			self.move_base_client.send_goal(goal)
			self.state_machine = 1


	def object_search_bruteforce(self, request):
		bl_pose = PoseStamped() 
		bl_pose.header.frame_id = "base_link" 
		bl_pose.pose.orientation.w = 1.0
		bl_pose.pose.position.x = 0.0
		bl_pose.pose.position.y = 0.0

		transform = self.tfBuffer.lookup_transform("map", "base_link", rospy.Time(0), rospy.Duration(1.0))
			
		bot_in_world = tf2_geometry_msgs.do_transform_pose(bl_pose, transform) #get current location of the robot in map frame 
			
		self.rooms_to_search = []
		
		room_dict = {'xmin':None, 'xmax':None, 'ymin':None, 'ymax':None, 'xcenter':None, 'ycenter':None, 'distance':None, 'scene_ids':[], 'confidences':[]}
		response = self.plc_in_map_srv(0.90)

		self.rooms_to_search = []
		for room in response.places: 
			room_dict['xmin'] = room.xmin
			room_dict['xmax'] = room.xmax
			room_dict['ymin'] = room.ymin
			room_dict['ymax'] = room.ymax
			room_dict['xcenter'] = (room.xmax + room.xmin)/2
			room_dict['ycenter'] = (room.ymax + room.ymin)/2
			room_dict['distance'] = np.sqrt( np.power(bot_in_world.pose.position.x - room_dict['xcenter'],2) + np.power(bot_in_world.pose.position.y - room_dict['ycenter'], 2) )
			room_dict['scene_ids'] = room.scene_ids
			room_dict['confidences'] = room.confidences
			self.rooms_to_search.append(deepcopy(room_dict))

		self.rooms_to_search = sorted(self.rooms_to_search, key=lambda k:k['distance'])
		self.marker_handler(action = 0, x = None, y = None)
		self.marker_handler(action = 2, x = None, y = None)

		print 'init_goals:{}'.format(self.rooms_to_search)


		print colored('Moving to x:{:.2f}, y{:.2f}','green').format(self.rooms_to_search[0]['xcenter'],self.rooms_to_search[0]['ycenter'])
		self.travelling_timer = time.time()
		goal = MoveBaseGoal()
		goal.target_pose.header.frame_id = "map"
		goal.target_pose.header.stamp = rospy.Time.now()
		goal.target_pose.pose.position.x = self.rooms_to_search[0]['xcenter']
		goal.target_pose.pose.position.y = self.rooms_to_search[0]['ycenter']
		goal.target_pose.pose.orientation.w = 1.0
		self.move_base_client.send_goal(goal)
		self.state_machine = 1



	def mb_status_cb(self,msg):

		if ((self.state_machine == 1) or (self.state_machine == 2)):
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
			orientation_list = (bot_in_world.pose.orientation.x,bot_in_world.pose.orientation.y,bot_in_world.pose.orientation.w,bot_in_world.pose.orientation.z)
			(roll,pitch,yaw) = euler_from_quaternion(orientation_list)
			
			px = bot_in_world.pose.position.x
			py = bot_in_world.pose.position.y

			distance = np.power(np.power(self.rooms_to_search[0]['xcenter']-px, 2) + np.power(self.rooms_to_search[0]['ycenter']-py,2),0.5)

			if self.state_machine == 1:
				if distance < self.min_goal_distance:
					cancel_goalid = actionlib.GoalID()
					self.cancel_pub.publish(cancel_goalid)
					self.state_machine = 2
					print colored('In a goal Position','green')
					self.init_rotation_timer = False

			if self.state_machine == 2:#robot rotating
				if self.init_rotation_timer == False:
					print colored('Start rotating for obtaining better place representation','green')
					vel_msg = Twist()
					vel_msg.linear.x = 0.0
					vel_msg.linear.y = 0.0
					vel_msg.linear.z = 0.0
					vel_msg.angular.x = 0.0
					vel_msg.angular.y = 0.0
					vel_msg.angular.z = np.pi * 2/41.79
					self.velocity_publisher.publish(vel_msg)
					self.init_rotation_timer = time.time()

				else:
					if (time.time()-self.init_rotation_timer) < 41.79:
						vel_msg = Twist()
						vel_msg.linear.x = 0.0
						vel_msg.linear.y = 0.0
						vel_msg.linear.z = 0.0
						vel_msg.angular.x = 0.0
						vel_msg.angular.y = 0.0
						vel_msg.angular.z = np.pi * 2/41.79
						self.velocity_publisher.publish(vel_msg)

					else:
						flag = False
						got_it_at = []
						for poses in self.obj_poses:
							if ( (poses[0] < self.rooms_to_search[0]['xmax']) and (poses[0] > self.rooms_to_search[0]['xmin']) and 
								 (poses[1] < self.rooms_to_search[0]['ymax']) and (poses[1] > self.rooms_to_search[0]['ymin']) ):
								end_timer = time.time()
								self.marker_handler(action = 1, x = poses[0], y = poses[1]) 
								got_it_at = [ poses[0], poses[1]]
								flag = True

						if flag == True:
							print self.rooms_to_search
							print self._PlacesConcepts[self.rooms_to_search[0]['scene_ids'][0]]
							print got_it_at[0]
							print got_it_at[1]
							print colored('Found the desired object in a {} at x{}:, y:{}!','green').format(self._PlacesConcepts[self.rooms_to_search[0]['scene_ids'][0]],got_it_at[0],got_it_at[1])
							print colored('Took {}sec ({}:{})','cyan').format(end_timer-self.travelling_timer, int((end_timer-self.travelling_timer)/60),int((end_timer-self.travelling_timer)%60))
							vel_msg = Twist()
							vel_msg.linear.x = 0.0
							vel_msg.linear.y = 0.0
							vel_msg.linear.z = 0.0
							vel_msg.angular.x = 0.0
							vel_msg.angular.y = 0.0
							vel_msg.angular.z = 0.0
							self.velocity_publisher.publish(vel_msg)
							self.init_rotation_timer = False
							self.rooms_to_search = []
							self.state_machine = 0
								
						else: 
							vel_msg = Twist()
							vel_msg.linear.x = 0.0
							vel_msg.linear.y = 0.0
							vel_msg.linear.z = 0.0
							vel_msg.angular.x = 0.0
							vel_msg.angular.y = 0.0
							vel_msg.angular.z = 0.0
							self.velocity_publisher.publish(vel_msg)
							self.init_rotation_timer = False
							
							self.marker_handler(action = 3, x = self.rooms_to_search[0]['xcenter'], y = self.rooms_to_search[0]['ycenter'])
							del self.rooms_to_search[0]

							if len(self.rooms_to_search) > 0:
								for k in xrange(0,len(self.rooms_to_search)):
									self.rooms_to_search[k]['distance'] = np.power(np.power(self.rooms_to_search[k]['xcenter']-px, 2) + np.power(self.rooms_to_search[k]['ycenter']-py,2),0.5)

								self.rooms_to_search = sorted(self.rooms_to_search, key=lambda k:k['distance'])
								self.marker_handler(action = 4, x = self.rooms_to_search[0]['xcenter'], y = self.rooms_to_search[0]['ycenter'])

								goal = MoveBaseGoal()
								self.state_machine = 1
								goal.target_pose.header.frame_id = "map"
								goal.target_pose.header.stamp = rospy.Time.now()
								goal.target_pose.pose.position.x = self.rooms_to_search[0]['xcenter']
								goal.target_pose.pose.position.y = self.rooms_to_search[0]['ycenter']
								goal.target_pose.pose.orientation.w = 1.0
								self.move_base_client.send_goal(goal)

							else:
								self.state_machine = 0
								print colored ('DID NOT FOUND THE REQUIRED OBJECT IN ANY OF THE DESIRED PLACES','red')


def main():

	rospy.init_node('object_finder_node')
	rospack = rospkg.RosPack()

	Objects_concepts = [ 'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 
								 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 
								 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
								 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 
								 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tv/monitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
								 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush' ]

	Places_concepts = [ 'abbey', 'airport_terminal', 'alley', 'amphitheater', 'amusement_park', 'aquarium', 'aqueduct', 'arch', 'art_gallery', 'art_studio', 'assembly_line', 
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

	SuperObject_concepts = [ 'vehicles', 'animals', 'food', 'electronics' ]

	places_in_map_service = rospy.get_param('~places_service', default = '/scene_mapper_node/scene_cat_srv')
	obj_finder_srv_namespace = rospy.get_param('~obj_finder_srv_namespace', default = '~obj_finder_srv')


	SuperObjects_Objects_Relations = np.load(os.path.join(rospack.get_path('reasoning_objects_places'),'files',rospy.get_param('~SupObjObj_file', default = 'SuperObjects_Objects_Relations.npy')))
	#print np.count_nonzero(SuperObjects_Objects_Relations)
	Object_LocatedAt_Place_Relations = np.load(os.path.join(rospack.get_path('reasoning_objects_places'),'files',rospy.get_param('~ObjPlace_file', default = 'Object_LocatedAt_Place_Relations.npy')))
	#print np.count_nonzero(Object_LocatedAt_Place_Relations)
	Object_NearBy_Object_Relations =  np.load(os.path.join(rospack.get_path('reasoning_objects_places'),'files',rospy.get_param('~ObjObj_file', default = 'Object_NearBy_Object_Relations.npy')))
	#print np.count_nonzero(Object_NearBy_Object_Relations)

	obj_label = "bed"
	obj_poses = [[-17.45,-5.46]]

	

	
	obj_searcher = ObjectSearcher(obj_label = obj_label, obj_poses = obj_poses, reasoning_algorithm = False, plc_in_map_srv = places_in_map_service, obj_finder_srv_namespace = obj_finder_srv_namespace,  
								  ObjConcepts = Objects_concepts, PlacesConcepts = Places_concepts, SuperObjectConcepts = SuperObject_concepts, 
								  pub_markers_topic = '~reasoning_markers' , SuperObjObj = SuperObjects_Objects_Relations, ObjPlace = Object_LocatedAt_Place_Relations, ObjObj = Object_NearBy_Object_Relations)
	
	rospy.spin()



'''
obj_label = 'sink'

	places_related = set( np.argwhere( Object_LocatedAt_Place_Relations[Objects_concepts.index(obj_label), :] ).ravel())
	if len(places_related) > 0:
		print colored(obj_label,'cyan'), ' can be ', colored('AtLocation: ','blue'), colored([Places_concepts[place] for place in places_related],'green')

	objects_related_from_nearby = np.argwhere( Object_NearBy_Object_Relations[Objects_concepts.index(obj_label), :] ).ravel()
	plcs = []
	
	if len(objects_related_from_nearby) > 0:
		print colored(obj_label, 'cyan'), ' can be found ', colored('NearBy: ','blue'), colored([Objects_concepts[obj] for obj in objects_related_from_nearby], 'green')
		for obj_related in objects_related_from_nearby:
			nb_places_related = np.argwhere(Object_LocatedAt_Place_Relations[obj_related, :]).ravel()
				
			if len(nb_places_related) > 0:
				print colored(Objects_concepts[obj_related],'magenta'), colored(' AtLocation: ','blue'), colored([Places_concepts[plc] for plc in nb_places_related],'green')
				for elem in nb_places_related:
					plcs.append(elem)
			
		if len(plcs) > 0:
			places_set = set()
			places_add = places_set.add
			plcs = [x for x in plcs if not (x in places_set or places_add(x))]
			print 'Therefore ', colored(obj_label,'cyan'), 'might also be ', colored('AtLocation: ','blue'), colored([Places_concepts[plc] for plc in plcs], 'green')

				
	if ( (len(objects_related_from_nearby) == 0) and (len(places_related) == 0) ):
		print colored( 'I do not where such object could be located', 'red')

	
	obj_label = 'apple'

	places_related = set( np.argwhere( Object_LocatedAt_Place_Relations[Objects_concepts.index(obj_label), :] ).ravel())
	if len(places_related) > 0:
		print colored(obj_label,'cyan'), ' can be ', colored('AtLocation: ','blue'), colored([Places_concepts[place] for place in places_related],'green')

	objects_related_from_nearby = np.argwhere( Object_NearBy_Object_Relations[Objects_concepts.index(obj_label), :] ).ravel()
	plcs = []
	
	if len(objects_related_from_nearby) > 0:
		print colored(obj_label, 'cyan'), ' can be found ', colored('NearBy: ','blue'), colored([Objects_concepts[obj] for obj in objects_related_from_nearby], 'green')
		for obj_related in objects_related_from_nearby:
			nb_places_related = np.argwhere(Object_LocatedAt_Place_Relations[obj_related, :]).ravel()
				
			if len(nb_places_related) > 0:
				print colored(Objects_concepts[obj_related],'magenta'), colored(' AtLocation: ','blue'), colored([Places_concepts[plc] for plc in nb_places_related],'green')
				for elem in nb_places_related:
					plcs.append(elem)
			
		if len(plcs) > 0:
			places_set = set()
			places_add = places_set.add
			plcs = [x for x in plcs if not (x in places_set or places_add(x))]
			print 'Therefore ', colored(obj_label,'cyan'), 'might also be ', colored('AtLocation: ','blue'), colored([Places_concepts[plc] for plc in plcs], 'green')

				
	if ( (len(objects_related_from_nearby) == 0) and (len(places_related) == 0) ):
		print colored( 'I do not where such object could be located', 'red')


	obj_label = 'tv/monitor'

	places_related = set( np.argwhere( Object_LocatedAt_Place_Relations[Objects_concepts.index(obj_label), :] ).ravel())
	if len(places_related) > 0:
		print colored(obj_label,'cyan'), ' can be ', colored('AtLocation: ','blue'), colored([Places_concepts[place] for place in places_related],'green')

	objects_related_from_nearby = np.argwhere( Object_NearBy_Object_Relations[Objects_concepts.index(obj_label), :] ).ravel()
	plcs = []
	
	if len(objects_related_from_nearby) > 0:
		print colored(obj_label, 'cyan'), ' can be found ', colored('NearBy: ','blue'), colored([Objects_concepts[obj] for obj in objects_related_from_nearby], 'green')
		for obj_related in objects_related_from_nearby:
			nb_places_related = np.argwhere(Object_LocatedAt_Place_Relations[obj_related, :]).ravel()
				
			if len(nb_places_related) > 0:
				print colored(Objects_concepts[obj_related],'magenta'), colored(' AtLocation: ','blue'), colored([Places_concepts[plc] for plc in nb_places_related],'green')
				for elem in nb_places_related:
					plcs.append(elem)
			
		if len(plcs) > 0:
			places_set = set()
			places_add = places_set.add
			plcs = [x for x in plcs if not (x in places_set or places_add(x))]
			print 'Therefore ', colored(obj_label,'cyan'), 'might also be ', colored('AtLocation: ','blue'), colored([Places_concepts[plc] for plc in plcs], 'green')

				
	if ( (len(objects_related_from_nearby) == 0) and (len(places_related) == 0) ):
		print colored( 'I do not where such object could be located', 'red')


	obj_label = 'bed'

	places_related = set( np.argwhere( Object_LocatedAt_Place_Relations[Objects_concepts.index(obj_label), :] ).ravel())
	if len(places_related) > 0:
		print colored(obj_label,'cyan'), ' can be ', colored('AtLocation: ','blue'), colored([Places_concepts[place] for place in places_related],'green')

	objects_related_from_nearby = np.argwhere( Object_NearBy_Object_Relations[Objects_concepts.index(obj_label), :] ).ravel()
	plcs = []
	
	if len(objects_related_from_nearby) > 0:
		print colored(obj_label, 'cyan'), ' can be found ', colored('NearBy: ','blue'), colored([Objects_concepts[obj] for obj in objects_related_from_nearby], 'green')
		for obj_related in objects_related_from_nearby:
			nb_places_related = np.argwhere(Object_LocatedAt_Place_Relations[obj_related, :]).ravel()
				
			if len(nb_places_related) > 0:
				print colored(Objects_concepts[obj_related],'magenta'), colored(' AtLocation: ','blue'), colored([Places_concepts[plc] for plc in nb_places_related],'green')
				for elem in nb_places_related:
					plcs.append(elem)
			
		if len(plcs) > 0:
			places_set = set()
			places_add = places_set.add
			plcs = [x for x in plcs if not (x in places_set or places_add(x))]
			print 'Therefore ', colored(obj_label,'cyan'), 'might also be ', colored('AtLocation: ','blue'), colored([Places_concepts[plc] for plc in plcs], 'green')

				
	if ( (len(objects_related_from_nearby) == 0) and (len(places_related) == 0) ):
		print colored( 'I do not where such object could be located', 'red')


	obj_label = 'fork'

	places_related = set( np.argwhere( Object_LocatedAt_Place_Relations[Objects_concepts.index(obj_label), :] ).ravel())
	if len(places_related) > 0:
		print colored(obj_label,'cyan'), ' can be ', colored('AtLocation: ','blue'), colored([Places_concepts[place] for place in places_related],'green')

	objects_related_from_nearby = np.argwhere( Object_NearBy_Object_Relations[Objects_concepts.index(obj_label), :] ).ravel()
	plcs = []
	
	if len(objects_related_from_nearby) > 0:
		print colored(obj_label, 'cyan'), ' can be found ', colored('NearBy: ','blue'), colored([Objects_concepts[obj] for obj in objects_related_from_nearby], 'green')
		for obj_related in objects_related_from_nearby:
			nb_places_related = np.argwhere(Object_LocatedAt_Place_Relations[obj_related, :]).ravel()
				
			if len(nb_places_related) > 0:
				print colored(Objects_concepts[obj_related],'magenta'), colored(' AtLocation: ','blue'), colored([Places_concepts[plc] for plc in nb_places_related],'green')
				for elem in nb_places_related:
					plcs.append(elem)
			
		if len(plcs) > 0:
			places_set = set()
			places_add = places_set.add
			plcs = [x for x in plcs if not (x in places_set or places_add(x))]
			print 'Therefore ', colored(obj_label,'cyan'), 'might also be ', colored('AtLocation: ','blue'), colored([Places_concepts[plc] for plc in plcs], 'green')

				
	if ( (len(objects_related_from_nearby) == 0) and (len(places_related) == 0) ):
		print colored( 'I do not where such object could be located', 'red')


	obj_label = 'keyboard'

	places_related = set( np.argwhere( Object_LocatedAt_Place_Relations[Objects_concepts.index(obj_label), :] ).ravel())
	if len(places_related) > 0:
		print colored(obj_label,'cyan'), ' can be ', colored('AtLocation: ','blue'), colored([Places_concepts[place] for place in places_related],'green')

	objects_related_from_nearby = np.argwhere( Object_NearBy_Object_Relations[Objects_concepts.index(obj_label), :] ).ravel()
	plcs = []
	
	if len(objects_related_from_nearby) > 0:
		print colored(obj_label, 'cyan'), ' can be found ', colored('NearBy: ','blue'), colored([Objects_concepts[obj] for obj in objects_related_from_nearby], 'green')
		for obj_related in objects_related_from_nearby:
			nb_places_related = np.argwhere(Object_LocatedAt_Place_Relations[obj_related, :]).ravel()
				
			if len(nb_places_related) > 0:
				print colored(Objects_concepts[obj_related],'magenta'), colored(' AtLocation: ','blue'), colored([Places_concepts[plc] for plc in nb_places_related],'green')
				for elem in nb_places_related:
					plcs.append(elem)
			
		if len(plcs) > 0:
			places_set = set()
			places_add = places_set.add
			plcs = [x for x in plcs if not (x in places_set or places_add(x))]
			print 'Therefore ', colored(obj_label,'cyan'), 'might also be ', colored('AtLocation: ','blue'), colored([Places_concepts[plc] for plc in plcs], 'green')

				
	if ( (len(objects_related_from_nearby) == 0) and (len(places_related) == 0) ):
		print colored( 'I do not where such object could be located', 'red')

	obj_label = 'toothbrush'
	obj_idx = Objects_concepts.index('toothbrush')
	plc_idx = Places_concepts.index('shower')
	print obj_idx
	print plc_idx
	print Object_LocatedAt_Place_Relations.shape
	print Object_LocatedAt_Place_Relations[obj_idx].shape
	Object_LocatedAt_Place_Relations[obj_idx][plc_idx] = True
	np.save(os.path.join(rospack.get_path('reasoning_objects_places'),'files',rospy.get_param('~ObjPlace_file', default = 'Object_LocatedAt_Place_Relations.npy')), Object_LocatedAt_Place_Relations)

	obj_label = 'toothbrush'
	places_related = set( np.argwhere( Object_LocatedAt_Place_Relations[Objects_concepts.index(obj_label), :] ).ravel())
	if len(places_related) > 0:
		print colored(obj_label,'cyan'), ' can be ', colored('AtLocation: ','blue'), colored([Places_concepts[place] for place in places_related],'green')

	objects_related_from_nearby = np.argwhere( Object_NearBy_Object_Relations[Objects_concepts.index(obj_label), :] ).ravel()
	plcs = []
	
	if len(objects_related_from_nearby) > 0:
		print colored(obj_label, 'cyan'), ' can be found ', colored('NearBy: ','blue'), colored([Objects_concepts[obj] for obj in objects_related_from_nearby], 'green')
		for obj_related in objects_related_from_nearby:
			nb_places_related = np.argwhere(Object_LocatedAt_Place_Relations[obj_related, :]).ravel()
				
			if len(nb_places_related) > 0:
				print colored(Objects_concepts[obj_related],'magenta'), colored(' AtLocation: ','blue'), colored([Places_concepts[plc] for plc in nb_places_related],'green')
				for elem in nb_places_related:
					plcs.append(elem)
			
		if len(plcs) > 0:
			places_set = set()
			places_add = places_set.add
			plcs = [x for x in plcs if not (x in places_set or places_add(x))]
			print 'Therefore ', colored(obj_label,'cyan'), 'might also be ', colored('AtLocation: ','blue'), colored([Places_concepts[plc] for plc in plcs], 'green')

				
	if ( (len(objects_related_from_nearby) == 0) and (len(places_related) == 0) ):
		print colored( 'I do not where such object could be located', 'red')


	obj_label = 'hair drier'
	obj_idx = Objects_concepts.index('hair drier')
	plc1_idx = Places_concepts.index('shower')
	plc2_idx = Places_concepts.index('bedroom')
	print obj_idx
	print plc1_idx
	print plc2_idx
	Object_LocatedAt_Place_Relations[obj_idx][[plc1_idx,plc2_idx]] = True
	np.save(os.path.join(rospack.get_path('reasoning_objects_places'),'files',rospy.get_param('~ObjPlace_file', default = 'Object_LocatedAt_Place_Relations.npy')), Object_LocatedAt_Place_Relations)
	
	obj_label = 'hair drier'
	places_related = set( np.argwhere( Object_LocatedAt_Place_Relations[Objects_concepts.index(obj_label), :] ).ravel())
	if len(places_related) > 0:
		print colored(obj_label,'cyan'), ' can be ', colored('AtLocation: ','blue'), colored([Places_concepts[place] for place in places_related],'green')

	objects_related_from_nearby = np.argwhere( Object_NearBy_Object_Relations[Objects_concepts.index(obj_label), :] ).ravel()
	plcs = []
	
	if len(objects_related_from_nearby) > 0:
		print colored(obj_label, 'cyan'), ' can be found ', colored('NearBy: ','blue'), colored([Objects_concepts[obj] for obj in objects_related_from_nearby], 'green')
		for obj_related in objects_related_from_nearby:
			nb_places_related = np.argwhere(Object_LocatedAt_Place_Relations[obj_related, :]).ravel()
				
			if len(nb_places_related) > 0:
				print colored(Objects_concepts[obj_related],'magenta'), colored(' AtLocation: ','blue'), colored([Places_concepts[plc] for plc in nb_places_related],'green')
				for elem in nb_places_related:
					plcs.append(elem)
			
		if len(plcs) > 0:
			places_set = set()
			places_add = places_set.add
			plcs = [x for x in plcs if not (x in places_set or places_add(x))]
			print 'Therefore ', colored(obj_label,'cyan'), 'might also be ', colored('AtLocation: ','blue'), colored([Places_concepts[plc] for plc in plcs], 'green')

				
	if ( (len(objects_related_from_nearby) == 0) and (len(places_related) == 0) ):
		print colored( 'I do not where such object could be located', 'red')



	obj_label = 'toaster'
	places_related = set( np.argwhere( Object_LocatedAt_Place_Relations[Objects_concepts.index(obj_label), :] ).ravel())
	if len(places_related) > 0:
		print colored(obj_label,'cyan'), ' can be ', colored('AtLocation: ','blue'), colored([Places_concepts[place] for place in places_related],'green')

	objects_related_from_nearby = np.argwhere( Object_NearBy_Object_Relations[Objects_concepts.index(obj_label), :] ).ravel()
	plcs = []
	
	if len(objects_related_from_nearby) > 0:
		print colored(obj_label, 'cyan'), ' can be found ', colored('NearBy: ','blue'), colored([Objects_concepts[obj] for obj in objects_related_from_nearby], 'green')
		for obj_related in objects_related_from_nearby:
			nb_places_related = np.argwhere(Object_LocatedAt_Place_Relations[obj_related, :]).ravel()
				
			if len(nb_places_related) > 0:
				print colored(Objects_concepts[obj_related],'magenta'), colored(' AtLocation: ','blue'), colored([Places_concepts[plc] for plc in nb_places_related],'green')
				for elem in nb_places_related:
					plcs.append(elem)
			
		if len(plcs) > 0:
			places_set = set()
			places_add = places_set.add
			plcs = [x for x in plcs if not (x in places_set or places_add(x))]
			print 'Therefore ', colored(obj_label,'cyan'), 'might also be ', colored('AtLocation: ','blue'), colored([Places_concepts[plc] for plc in plcs], 'green')

				
	if ( (len(objects_related_from_nearby) == 0) and (len(places_related) == 0) ):
		print colored( 'I do not where such object could be located', 'red')


	obj_label = 'mouse'
	places_related = set( np.argwhere( Object_LocatedAt_Place_Relations[Objects_concepts.index(obj_label), :] ).ravel())
	if len(places_related) > 0:
		print colored(obj_label,'cyan'), ' can be ', colored('AtLocation: ','blue'), colored([Places_concepts[place] for place in places_related],'green')

	objects_related_from_nearby = np.argwhere( Object_NearBy_Object_Relations[Objects_concepts.index(obj_label), :] ).ravel()
	plcs = []
	
	if len(objects_related_from_nearby) > 0:
		print colored(obj_label, 'cyan'), ' can be found ', colored('NearBy: ','blue'), colored([Objects_concepts[obj] for obj in objects_related_from_nearby], 'green')
		for obj_related in objects_related_from_nearby:
			nb_places_related = np.argwhere(Object_LocatedAt_Place_Relations[obj_related, :]).ravel()
				
			if len(nb_places_related) > 0:
				print colored(Objects_concepts[obj_related],'magenta'), colored(' AtLocation: ','blue'), colored([Places_concepts[plc] for plc in nb_places_related],'green')
				for elem in nb_places_related:
					plcs.append(elem)
			
		if len(plcs) > 0:
			places_set = set()
			places_add = places_set.add
			plcs = [x for x in plcs if not (x in places_set or places_add(x))]
			print 'Therefore ', colored(obj_label,'cyan'), 'might also be ', colored('AtLocation: ','blue'), colored([Places_concepts[plc] for plc in plcs], 'green')

				
	if ( (len(objects_related_from_nearby) == 0) and (len(places_related) == 0) ):
		print colored( 'I do not where such object could be located', 'red')

	obj_label = 'scissors'
	obj_idx = Objects_concepts.index('scissors')
	plc_idx = [Places_concepts.index('classroom'), Places_concepts.index('auditorium'), Places_concepts.index('home_office'), Places_concepts.index('office')]
	print obj_idx
	print plc
	Object_LocatedAt_Place_Relations[obj_idx][plc_idx] = True
	np.save(os.path.join(rospack.get_path('reasoning_objects_places'),'files',rospy.get_param('~ObjPlace_file', default = 'Object_LocatedAt_Place_Relations.npy')), Object_LocatedAt_Place_Relations)

	obj_label = 'scissors'
	places_related = set( np.argwhere( Object_LocatedAt_Place_Relations[Objects_concepts.index(obj_label), :] ).ravel())
	if len(places_related) > 0:
		print colored(obj_label,'cyan'), ' can be ', colored('AtLocation: ','blue'), colored([Places_concepts[place] for place in places_related],'green')

	objects_related_from_nearby = np.argwhere( Object_NearBy_Object_Relations[Objects_concepts.index(obj_label), :] ).ravel()
	plcs = []
	
	if len(objects_related_from_nearby) > 0:
		print colored(obj_label, 'cyan'), ' can be found ', colored('NearBy: ','blue'), colored([Objects_concepts[obj] for obj in objects_related_from_nearby], 'green')
		for obj_related in objects_related_from_nearby:
			nb_places_related = np.argwhere(Object_LocatedAt_Place_Relations[obj_related, :]).ravel()
				
			if len(nb_places_related) > 0:
				print colored(Objects_concepts[obj_related],'magenta'), colored(' AtLocation: ','blue'), colored([Places_concepts[plc] for plc in nb_places_related],'green')
				for elem in nb_places_related:
					plcs.append(elem)
			
		if len(plcs) > 0:
			places_set = set()
			places_add = places_set.add
			plcs = [x for x in plcs if not (x in places_set or places_add(x))]
			print 'Therefore ', colored(obj_label,'cyan'), 'might also be ', colored('AtLocation: ','blue'), colored([Places_concepts[plc] for plc in plcs], 'green')

				
	if ( (len(objects_related_from_nearby) == 0) and (len(places_related) == 0) ):
		print colored( 'I do not where such object could be located', 'red')


	obj_label = 'oven'
	places_related = set( np.argwhere( Object_LocatedAt_Place_Relations[Objects_concepts.index(obj_label), :] ).ravel())
	if len(places_related) > 0:
		print colored(obj_label,'cyan'), ' can be ', colored('AtLocation: ','blue'), colored([Places_concepts[place] for place in places_related],'green')

	objects_related_from_nearby = np.argwhere( Object_NearBy_Object_Relations[Objects_concepts.index(obj_label), :] ).ravel()
	plcs = []
	
	if len(objects_related_from_nearby) > 0:
		print colored(obj_label, 'cyan'), ' can be found ', colored('NearBy: ','blue'), colored([Objects_concepts[obj] for obj in objects_related_from_nearby], 'green')
		for obj_related in objects_related_from_nearby:
			nb_places_related = np.argwhere(Object_LocatedAt_Place_Relations[obj_related, :]).ravel()
				
			if len(nb_places_related) > 0:
				print colored(Objects_concepts[obj_related],'magenta'), colored(' AtLocation: ','blue'), colored([Places_concepts[plc] for plc in nb_places_related],'green')
				for elem in nb_places_related:
					plcs.append(elem)
			
		if len(plcs) > 0:
			places_set = set()
			places_add = places_set.add
			plcs = [x for x in plcs if not (x in places_set or places_add(x))]
			print 'Therefore ', colored(obj_label,'cyan'), 'might also be ', colored('AtLocation: ','blue'), colored([Places_concepts[plc] for plc in plcs], 'green')

				
	if ( (len(objects_related_from_nearby) == 0) and (len(places_related) == 0) ):
		print colored( 'I do not where such object could be located', 'red')




	obj_label = 'laptop'
	places_related = set( np.argwhere( Object_LocatedAt_Place_Relations[Objects_concepts.index(obj_label), :] ).ravel())
	if len(places_related) > 0:
		print colored(obj_label,'cyan'), ' can be ', colored('AtLocation: ','blue'), colored([Places_concepts[place] for place in places_related],'green')

	objects_related_from_nearby = np.argwhere( Object_NearBy_Object_Relations[Objects_concepts.index(obj_label), :] ).ravel()
	plcs = []
	
	if len(objects_related_from_nearby) > 0:
		print colored(obj_label, 'cyan'), ' can be found ', colored('NearBy: ','blue'), colored([Objects_concepts[obj] for obj in objects_related_from_nearby], 'green')
		for obj_related in objects_related_from_nearby:
			nb_places_related = np.argwhere(Object_LocatedAt_Place_Relations[obj_related, :]).ravel()
				
			if len(nb_places_related) > 0:
				print colored(Objects_concepts[obj_related],'magenta'), colored(' AtLocation: ','blue'), colored([Places_concepts[plc] for plc in nb_places_related],'green')
				for elem in nb_places_related:
					plcs.append(elem)
			
		if len(plcs) > 0:
			places_set = set()
			places_add = places_set.add
			plcs = [x for x in plcs if not (x in places_set or places_add(x))]
			print 'Therefore ', colored(obj_label,'cyan'), 'might also be ', colored('AtLocation: ','blue'), colored([Places_concepts[plc] for plc in plcs], 'green')

				
	if ( (len(objects_related_from_nearby) == 0) and (len(places_related) == 0) ):
		print colored( 'I do not where such object could be located', 'red')

	
	obj_label = 'pizza'
	obj_idx = Objects_concepts.index('pizza')
	plcs_pizza = np.argwhere( Object_LocatedAt_Place_Relations[Objects_concepts.index(obj_label), :] ).ravel()
	print plcs_pizza
	print [Places_concepts[idx] for idx in plcs_pizza] 
	#Object_LocatedAt_Place_Relations[obj_idx][plcs_pizza] = False
	places_related = set( np.argwhere( Object_LocatedAt_Place_Relations[Objects_concepts.index(obj_label), :] ).ravel())
	#np.save(os.path.join(rospack.get_path('reasoning_objects_places'),'files',rospy.get_param('~ObjPlace_file', default = 'Object_LocatedAt_Place_Relations.npy')), Object_LocatedAt_Place_Relations)
	

	obj_label = 'pizza'
	places_related = set( np.argwhere( Object_LocatedAt_Place_Relations[Objects_concepts.index(obj_label), :] ).ravel())
	if len(places_related) > 0:
		print colored(obj_label,'cyan'), ' can be ', colored('AtLocation: ','blue'), colored([Places_concepts[place] for place in places_related],'green')

	objects_related_from_nearby = np.argwhere( Object_NearBy_Object_Relations[Objects_concepts.index(obj_label), :] ).ravel()
	plcs = []
	
	if len(objects_related_from_nearby) > 0:
		print colored(obj_label, 'cyan'), ' can be found ', colored('NearBy: ','blue'), colored([Objects_concepts[obj] for obj in objects_related_from_nearby], 'green')
		for obj_related in objects_related_from_nearby:
			nb_places_related = np.argwhere(Object_LocatedAt_Place_Relations[obj_related, :]).ravel()
				
			if len(nb_places_related) > 0:
				print colored(Objects_concepts[obj_related],'magenta'), colored(' AtLocation: ','blue'), colored([Places_concepts[plc] for plc in nb_places_related],'green')
				for elem in nb_places_related:
					plcs.append(elem)
			
		if len(plcs) > 0:
			places_set = set()
			places_add = places_set.add
			plcs = [x for x in plcs if not (x in places_set or places_add(x))]
			print 'Therefore ', colored(obj_label,'cyan'), 'might also be ', colored('AtLocation: ','blue'), colored([Places_concepts[plc] for plc in plcs], 'green')

				
	if ( (len(objects_related_from_nearby) == 0) and (len(places_related) == 0) ):
		print colored( 'I do not where such object could be located', 'red')

	
	obj_label = 'wine glass'
	places_related = set( np.argwhere( Object_LocatedAt_Place_Relations[Objects_concepts.index(obj_label), :] ).ravel())
	if len(places_related) > 0:
		print colored(obj_label,'cyan'), ' can be ', colored('AtLocation: ','blue'), colored([Places_concepts[place] for place in places_related],'green')

	objects_related_from_nearby = np.argwhere( Object_NearBy_Object_Relations[Objects_concepts.index(obj_label), :] ).ravel()
	plcs = []
	
	if len(objects_related_from_nearby) > 0:
		print colored(obj_label, 'cyan'), ' can be found ', colored('NearBy: ','blue'), colored([Objects_concepts[obj] for obj in objects_related_from_nearby], 'green')
		for obj_related in objects_related_from_nearby:
			nb_places_related = np.argwhere(Object_LocatedAt_Place_Relations[obj_related, :]).ravel()
				
			if len(nb_places_related) > 0:
				print colored(Objects_concepts[obj_related],'magenta'), colored(' AtLocation: ','blue'), colored([Places_concepts[plc] for plc in nb_places_related],'green')
				for elem in nb_places_related:
					plcs.append(elem)
			
		if len(plcs) > 0:
			places_set = set()
			places_add = places_set.add
			plcs = [x for x in plcs if not (x in places_set or places_add(x))]
			print 'Therefore ', colored(obj_label,'cyan'), 'might also be ', colored('AtLocation: ','blue'), colored([Places_concepts[plc] for plc in plcs], 'green')

				
	if ( (len(objects_related_from_nearby) == 0) and (len(places_related) == 0) ):
		print colored( 'I do not where such object could be located', 'red')


	
	obj_label = 'tie'
	obj_idx = Objects_concepts.index('tie')
	plc_idx = [Places_concepts.index('bedroom'), Places_concepts.index('closet')]
	print obj_idx
	print plc
	Object_LocatedAt_Place_Relations[obj_idx][plc_idx] = True
	np.save(os.path.join(rospack.get_path('reasoning_objects_places'),'files',rospy.get_param('~ObjPlace_file', default = 'Object_LocatedAt_Place_Relations.npy')), Object_LocatedAt_Place_Relations)
	

	obj_label = 'tie'
	places_related = set( np.argwhere( Object_LocatedAt_Place_Relations[Objects_concepts.index(obj_label), :] ).ravel())
	if len(places_related) > 0:
		print colored(obj_label,'cyan'), ' can be ', colored('AtLocation: ','blue'), colored([Places_concepts[place] for place in places_related],'green')

	objects_related_from_nearby = np.argwhere( Object_NearBy_Object_Relations[Objects_concepts.index(obj_label), :] ).ravel()
	plcs = []
	
	if len(objects_related_from_nearby) > 0:
		print colored(obj_label, 'cyan'), ' can be found ', colored('NearBy: ','blue'), colored([Objects_concepts[obj] for obj in objects_related_from_nearby], 'green')
		for obj_related in objects_related_from_nearby:
			nb_places_related = np.argwhere(Object_LocatedAt_Place_Relations[obj_related, :]).ravel()
				
			if len(nb_places_related) > 0:
				print colored(Objects_concepts[obj_related],'magenta'), colored(' AtLocation: ','blue'), colored([Places_concepts[plc] for plc in nb_places_related],'green')
				for elem in nb_places_related:
					plcs.append(elem)
			
		if len(plcs) > 0:
			places_set = set()
			places_add = places_set.add
			plcs = [x for x in plcs if not (x in places_set or places_add(x))]
			print 'Therefore ', colored(obj_label,'cyan'), 'might also be ', colored('AtLocation: ','blue'), colored([Places_concepts[plc] for plc in plcs], 'green')

				
	if ( (len(objects_related_from_nearby) == 0) and (len(places_related) == 0) ):
		print colored( 'I do not where such object could be located', 'red')


	obj_label = 'toilet'
	places_related = set( np.argwhere( Object_LocatedAt_Place_Relations[Objects_concepts.index(obj_label), :] ).ravel())
	if len(places_related) > 0:
		print colored(obj_label,'cyan'), ' can be ', colored('AtLocation: ','blue'), colored([Places_concepts[place] for place in places_related],'green')

	objects_related_from_nearby = np.argwhere( Object_NearBy_Object_Relations[Objects_concepts.index(obj_label), :] ).ravel()
	plcs = []
	
	if len(objects_related_from_nearby) > 0:
		print colored(obj_label, 'cyan'), ' can be found ', colored('NearBy: ','blue'), colored([Objects_concepts[obj] for obj in objects_related_from_nearby], 'green')
		for obj_related in objects_related_from_nearby:
			nb_places_related = np.argwhere(Object_LocatedAt_Place_Relations[obj_related, :]).ravel()
				
			if len(nb_places_related) > 0:
				print colored(Objects_concepts[obj_related],'magenta'), colored(' AtLocation: ','blue'), colored([Places_concepts[plc] for plc in nb_places_related],'green')
				for elem in nb_places_related:
					plcs.append(elem)
			
		if len(plcs) > 0:
			places_set = set()
			places_add = places_set.add
			plcs = [x for x in plcs if not (x in places_set or places_add(x))]
			print 'Therefore ', colored(obj_label,'cyan'), 'might also be ', colored('AtLocation: ','blue'), colored([Places_concepts[plc] for plc in plcs], 'green')

				
	if ( (len(objects_related_from_nearby) == 0) and (len(places_related) == 0) ):
		print colored( 'I do not where such object could be located', 'red')


	obj_label = 'cake'
	places_related = set( np.argwhere( Object_LocatedAt_Place_Relations[Objects_concepts.index(obj_label), :] ).ravel())
	if len(places_related) > 0:
		print colored(obj_label,'cyan'), ' can be ', colored('AtLocation: ','blue'), colored([Places_concepts[place] for place in places_related],'green')

	objects_related_from_nearby = np.argwhere( Object_NearBy_Object_Relations[Objects_concepts.index(obj_label), :] ).ravel()
	plcs = []
	
	if len(objects_related_from_nearby) > 0:
		print colored(obj_label, 'cyan'), ' can be found ', colored('NearBy: ','blue'), colored([Objects_concepts[obj] for obj in objects_related_from_nearby], 'green')
		for obj_related in objects_related_from_nearby:
			nb_places_related = np.argwhere(Object_LocatedAt_Place_Relations[obj_related, :]).ravel()
				
			if len(nb_places_related) > 0:
				print colored(Objects_concepts[obj_related],'magenta'), colored(' AtLocation: ','blue'), colored([Places_concepts[plc] for plc in nb_places_related],'green')
				for elem in nb_places_related:
					plcs.append(elem)
			
		if len(plcs) > 0:
			places_set = set()
			places_add = places_set.add
			plcs = [x for x in plcs if not (x in places_set or places_add(x))]
			print 'Therefore ', colored(obj_label,'cyan'), 'might also be ', colored('AtLocation: ','blue'), colored([Places_concepts[plc] for plc in plcs], 'green')

				
	if ( (len(objects_related_from_nearby) == 0) and (len(places_related) == 0) ):
		print colored( 'I do not where such object could be located', 'red')


	obj_label = 'sofa'
	places_related = set( np.argwhere( Object_LocatedAt_Place_Relations[Objects_concepts.index(obj_label), :] ).ravel())
	if len(places_related) > 0:
		print colored(obj_label,'cyan'), ' can be ', colored('AtLocation: ','blue'), colored([Places_concepts[place] for place in places_related],'green')

	objects_related_from_nearby = np.argwhere( Object_NearBy_Object_Relations[Objects_concepts.index(obj_label), :] ).ravel()
	plcs = []
	
	if len(objects_related_from_nearby) > 0:
		print colored(obj_label, 'cyan'), ' can be found ', colored('NearBy: ','blue'), colored([Objects_concepts[obj] for obj in objects_related_from_nearby], 'green')
		for obj_related in objects_related_from_nearby:
			nb_places_related = np.argwhere(Object_LocatedAt_Place_Relations[obj_related, :]).ravel()
				
			if len(nb_places_related) > 0:
				print colored(Objects_concepts[obj_related],'magenta'), colored(' AtLocation: ','blue'), colored([Places_concepts[plc] for plc in nb_places_related],'green')
				for elem in nb_places_related:
					plcs.append(elem)
			
		if len(plcs) > 0:
			places_set = set()
			places_add = places_set.add
			plcs = [x for x in plcs if not (x in places_set or places_add(x))]
			print 'Therefore ', colored(obj_label,'cyan'), 'might also be ', colored('AtLocation: ','blue'), colored([Places_concepts[plc] for plc in plcs], 'green')

				
	if ( (len(objects_related_from_nearby) == 0) and (len(places_related) == 0) ):
		print colored( 'I do not where such object could be located', 'red')


'''






