import rospy
import rospkg
from object_detection.msg import Detections
from sem_map.msg import ObjectInMap, Place
from sem_map.srv import ObjectInstances, ObjectInstancesResponse, places_map_server, places_map_serverResponse
from std_srvs.srv import Trigger, TriggerResponse
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Joy
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
	
	def __init__(self, obj_in_map_srv, plc_in_map_srv, obj_finder_srv_namespace, detections_topic, joystick_topic, 
				sub_pointcloud_topic, pub_markers_topic, ObjConcepts, PlacesConcepts, SuperObjectConcepts, SuperObjObj, ObjPlace, ObjObj):
		
		self._ObjConcepts = ObjConcepts #list of object labels
		self._PlacesConcepts = PlacesConcepts #list of places labels
		self._SuperObjectConcepts = SuperObjectConcepts #list of superconcepts labels [vehicles, animals, food, electronics]
		self._SuperObjObj = SuperObjObj #super objects to objects relations (e.g. apple is a food)
										#boolean np.array dim(4,80) (superconcept, object) describing if there is relation or not between the object and the super category
		self._ObjPlace = ObjPlace # boolean np.array describing the relations between objects and places (80,205) (AtLocationRelation)
		self._ObjObj = ObjObj #boolena numpy array describing NearbyRelation (80,80) apple Nearby Banana = TRue, therefore self._ObjObj[apple_idx, banana_idx] = True

		self.obj_finder_server = rospy.Service(obj_finder_srv_namespace, Trigger, self.object_search)
		
		self.min_scene_belief = 0.25
		self.min_goal_distance = 0.35

		self.tfBuffer = tf2_ros.Buffer()
		self.listener = tf2_ros.TransformListener(self.tfBuffer)

		self.state_machine = 0
		self.search_id = None
		self.ndetect = 0
		#state 0 -> waiting for searching request 
		#state 1 -> moving to a destination to search
		#state 2 -> stopped due to a navigation goal
		#state 3 -> rotating in a desired position 
		self.goals = []

		self.stop_counter = 0

		self.rotation_started = False
		self.init_rotation_timer = 0

		self.mb_status = rospy.Subscriber('move_base/status', actionlib.GoalStatusArray, self.mb_status_cb, queue_size = 1)

		
		self.joy_subscriber = rospy.Subscriber(joystick_topic, Joy, self.joystick_safety, queue_size = 1)

		self.obj_sub = rospy.Subscriber(detections_topic, Detections, self.detections_callback, queue_size = 1)

		self.cancel_pub = rospy.Publisher('move_base/cancel', actionlib.GoalID, queue_size = 10)
		#self._img_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.img_callback, queue_size=1, buff_size=2**24) #subscribe image topic
		self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size = 1)
		self.pc_xyz_sub = rospy.Subscriber(sub_pointcloud_topic, PointCloud2, self.pointcloud_callback, queue_size=1)
		self.marker_pub = rospy.Publisher(pub_markers_topic, Marker, queue_size = 200)

		self.move_base_client = actionlib.SimpleActionClient('move_base',MoveBaseAction) #move base serve to autonomously navigate
		print 'waiting for move_base server'
		self.move_base_client.wait_for_server()
		print 'Move base available'

		rospy.wait_for_service(obj_in_map_srv)
		try:
			self.obj_in_map_srv = rospy.ServiceProxy(obj_in_map_srv, ObjectInstances) #object's map service client
		except rospy.ServiceException, e:
			print "Service call failed: %s"%e

		print 'Object mapper server available'

		rospy.wait_for_service(plc_in_map_srv)
		try:
			self.plc_in_map_srv = rospy.ServiceProxy(plc_in_map_srv, places_map_server) #place's map service client
		except rospy.ServiceException, e:
			print "Service call failed: %s"%e

		print 'Scene mapper server available'


	def pointcloud_callback(self, pointcloud_msg):
		self.curr_pc = pointcloud_msg
	
	def get_object_3D_point_bb(self,width_min,width_max,height_min,height_max,histogram_method): #method to localize object after finding it
		
		bb_pts=((i,j) for i in xrange(width_min,width_max) for j in xrange(height_min, height_max))#pixels of the image to verify in the pointcloud
		points = np.array(list(pc2.read_points(self.curr_pc, field_names = ("x", "y", "z"), skip_nans=True, uvs=bb_pts)))

		if (points.size!=0):
			nan_perc = (1.0 - float(float(points.size/3)/float((width_max-width_min)*(height_max-height_min))))
			z_min = min(points[:,2]) #minimum distance to the objected acepted to annotate the object
			z_max = max(points[:,2]) #max depth value observed
				
			z_hist, bins = np.histogram(points[:,2], bins=histogram_method, range=(z_min,z_max)) #histogram of depth values
			z_points_bins = np.digitize(points[:,2],bins) #the indice of the bin for each point (0 is the first interval)
			z_mode, countss = mode(z_points_bins)
				
			points_indexes = [index for index, point_bin in enumerate(z_points_bins) if point_bin == z_mode]
			cam_position = np.mean(points[points_indexes,:],axis=0) #get the mean of x,y,z's  
			x_pos = cam_position[0]
			y_pos = cam_position[1]
			z_pos = cam_position[2]
			
			#fill the posestamped with the position variables, so that we can transform them to world later on 	
			depth_optical_frame_pose = PoseStamped() #pose stamped of the object point in the camera_depth_optical frame 
			depth_optical_frame_pose.header.frame_id = "camera_depth_optical_frame"
			depth_optical_frame_pose.pose.orientation.w = 1.0
		
			depth_optical_frame_pose.header.stamp = rospy.Time.now() 
			depth_optical_frame_pose.pose.position.x = x_pos
			depth_optical_frame_pose.pose.position.y = y_pos
			depth_optical_frame_pose.pose.position.z = z_pos

			dist = np.sqrt((depth_optical_frame_pose.pose.position.x**2)+(depth_optical_frame_pose.pose.position.y**2)+(depth_optical_frame_pose.pose.position.z**2))

			transform = self.tfBuffer.lookup_transform("map",
							"camera_depth_optical_frame", 
							rospy.Time(0),
							rospy.Duration(1.0)) #wait for 1 second

			pose_transformed = tf2_geometry_msgs.do_transform_pose(depth_optical_frame_pose, transform) #apply the transform to the object in the camera frame (get its world coordinates)
					#print 'time_to_transform{}'.format(time.time()-t
			rect_dict = {'x_pos':pose_transformed.pose.position.x,'y_pos':pose_transformed.pose.position.y,'z_pos':pose_transformed.pose.position.z, 'verf':True, 'nan_perc':nan_perc}
					#print 'passed location test'

			return rect_dict
		else:
			rect_dict = {'x_pos':0,'y_pos':0,'z_pos':0, 'verf':False}
					#print 'passed location test'
			return rect_dict


	def joystick_safety(self, msg):
		if ( (msg.buttons[2]) and (not msg.buttons[3]) ): #SAFETY -> (stop move_base navigation, stop current goal)
			if self.stop_counter == 0:
				print colored('Canceling Autonomous Navigation -> Teleoperate Now!','red')
				self.stop_counter = 1
			
			stop_goalid = actionlib.GoalID()
			self.cancel_pub.publish(stop_goalid)
				
			if self.state_machine != 0:
				self.state_machine = 2

		if ( (self.state_machine == 2) and (msg.buttons[3]) and (not msg.buttons[2]) ): #continue AUTONOMOUS navigation
				self.stop_counter = 0
				goal = MoveBaseGoal()
				goal.target_pose.header.frame_id = "map"
				goal.target_pose.header.stamp = rospy.Time.now()
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
			
				px = bot_in_world.pose.position.x
				py = bot_in_world.pose.position.y

				for k in xrange(0,len(self.goals)): #reorder navigation goals
					self.goals[k]['distance'] = np.power(np.power(self.goals[k]['xcenter']-px, 2) + np.power(self.goals[k]['ycenter']-py,2),0.5)
							
				self.goals = sorted(self.goals, key=lambda k:k['distance'])

				goal.target_pose.pose.position.x = self.goals[0]['xcenter']
				goal.target_pose.pose.position.y = self.goals[0]['ycenter']
				goal.target_pose.pose.orientation.w = 1.0
				
				self.move_base_client.send_goal(goal) #set goal to nearest localization where there is a high likelihood of finding the desired object
				print colored('Autonomous Navigation resumed -> Goal set to:[{:.2f},{:.2f}]','green').format(self.goals[0]['xcenter'],self.goals[0]['ycenter'])
				self.state_machine = 1

	def mb_status_cb(self,msg):
			
		if (self.state_machine == 1):
			mb_state = self.move_base_client.get_state()

			if (mb_state in [4,5]): #4 and 5 move base states mean that the goal set was denied and the robot will not move towards those (move_base failed in obtaining a safe trajectory to reach goal)
				#clear added markers as robot will no longer search object -> robot cannot move to its destination (move_base aborted its goal)
				for j in xrange(0,len(self.goals)):
					self.marker_handler(x = self.goals[j]['xcenter'], y = self.goals[j]['ycenter'], z = None , mk_type = 0, action = 1, label = None, current = False, final = False) #deleting marker goals
				#deleting goals
				self.obj_id = None
				self.state_machine = 0 #stop searching for object
				self.goals = []

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
				px = bot_in_world.pose.position.x
				py = bot_in_world.pose.position.y
				self.marker_handler(x = px, y = py, z = None , mk_type = 0, action = 2, label = None, current = False, final = False) #deleting marker goals
				print colored('Navigation failed! Cant search the desired object','red')


		if self.state_machine in [1,3]: #searching for object autonously or still searching but being teleoperated due to safety reasons
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

			distance = np.power(np.power(self.goals[0]['xcenter']-px, 2) + np.power(self.goals[0]['ycenter']-py,2),0.5) #distance from robot to current goal set to move_base

			if self.state_machine == 1:
				if mb_state == 3:
					cancel_goalid = actionlib.GoalID()
					self.cancel_pub.publish(cancel_goalid)
					self.state_machine = 3
					print colored('Near a Goal Position:[{:.2f},{:.2f}]','green').format(self.goals[0]['xcenter'],self.goals[0]['ycenter'])
					self.init_rotation_timer = 0

			if self.state_machine == 3:#robot reached a navigation goal (rotating now for obtaining a better representation of its location)
				if self.rotation_started == False:
					print colored('Started Rotation (Obtaining a better view)','green')
					vel_msg = Twist()
					vel_msg.linear.x = 0.0
					vel_msg.linear.y = 0.0
					vel_msg.linear.z = 0.0
					vel_msg.angular.x = 0.0
					vel_msg.angular.y = 0.0
					vel_msg.angular.z = np.pi*2/15
					self.velocity_publisher.publish(vel_msg)
					self.init_rotation_timer = time.time()
					self.rotation_started = True

				else:
					if (time.time()-self.init_rotation_timer) < 15.0: #perform a rotation (Sending angular velocities directly)
						vel_msg = Twist()
						vel_msg.linear.x = 0.0
						vel_msg.linear.y = 0.0
						vel_msg.linear.z = 0.0
						vel_msg.angular.x = 0.0
						vel_msg.angular.y = 0.0
						vel_msg.angular.z = np.pi * 2/15
						self.velocity_publisher.publish(vel_msg)

					else:
						print colored('Object not found near this goal','red') #rotation was already performed
						vel_msg = Twist()
						vel_msg.linear.x = 0.0
						vel_msg.linear.y = 0.0
						vel_msg.linear.z = 0.0
						vel_msg.angular.x = 0.0
						vel_msg.angular.y = 0.0
						vel_msg.angular.z = 0.0
						self.velocity_publisher.publish(vel_msg)
						self.rotation_started == False
						self.init_rotation_timer = 0

						if len(self.goals[0]) > 0:
							self.marker_handler(x = self.goals[0]['xcenter'], y = self.goals[0]['ycenter'], z = None , mk_type = 0, action = 1, label = None, current = False, final = False) #remove this location marker (it was already searched)
							del self.goals[0]
							for k in xrange(0,len(self.goals)): #reorder nav-goals given distance from those locations to the base_link of the robot 
								self.goals[k]['distance'] = np.power(np.power(self.goals[k]['xcenter']-px, 2) + np.power(self.goals[k]['ycenter']-py,2),0.5)
							
							self.goals = sorted(self.goals, key=lambda k:k['distance'])
							self.marker_handler(x = self.goals[0]['xcenter'], y = self.goals[0]['ycenter'], z = self.goals[0]['zcenter'] , mk_type = 0, action = 0, label = None, current = True, final = False)
							

						if len(self.goals) > 0:
							goal = MoveBaseGoal()
							self.state_machine = 1
							goal.target_pose.header.frame_id = "map"
							goal.target_pose.header.stamp = rospy.Time.now()
							goal.target_pose.pose.position.x = self.goals[0]['xcenter']
							goal.target_pose.pose.position.y = self.goals[0]['ycenter']
							goal.target_pose.pose.orientation.w = 1.0
							print colored('Next Goal was set to:[{:.2f},{:.2f}]','green').format(self.goals[0]['xcenter'],self.goals[0]['ycenter'])
							self.move_base_client.send_goal(goal)

						if len(self.goals) == 0:
							self.state_machine = 0
							print colored ('DID NOT FOUND THE REQUIRED OBJECT!!!','red')


	def detections_callback(self, msg):
		
		if self.state_machine != 0:
			same_objs = []
			label = self._ObjConcepts[self.search_id]
			
			for obj in msg.objects:
				if obj.id == self.search_id:
					same_objs.append(deepcopy(obj))

			if len(same_objs) > 0:
				for elem in same_objs:
					dic = self.get_object_3D_point_bb(elem.width_min,elem.width_max,elem.height_min,elem.height_max,'fd')
					bl_pose = PoseStamped() 
					bl_pose.header.frame_id = "camera_depth_optical_frame" 
					bl_pose.pose.orientation.w = 1.0
					bl_pose.pose.position.x = 0.0
					bl_pose.pose.position.y = 0.0

					transform = self.tfBuffer.lookup_transform("map",
									"camera_depth_optical_frame", 
									rospy.Time(0), #get the tf at first available time
									rospy.Duration(1.0)) #wait for 1 second

					cam_in_world = tf2_geometry_msgs.do_transform_pose(bl_pose, transform) #get current location of the robot in map frame 
					px = cam_in_world.pose.position.x
					py = cam_in_world.pose.position.y

					distance = np.power(np.power(dic['x_pos']-px, 2) + np.power(dic['y_pos']-py,2),0.5)

					if dic['verf'] == True and distance < 2.80 and dic['nan_perc'] < 0.65:
						self.marker_handler(x = dic['x_pos'] , y = dic['y_pos'] , z = dic['z_pos'], mk_type = 1, action = 0, label = elem.label, current = False, final = True)
						print 'Founded a {} at [x,y,z]=[{:.2f},{:.2f},{:.2f}]'.format(elem.label,dic['x_pos'],dic['y_pos'],dic['z_pos'])
						self.state_machine = 0
						
				
						cancel_goalid = actionlib.GoalID()
						self.cancel_pub.publish(cancel_goalid)

						vel_msg = Twist()
						vel_msg.linear.x = 0.0
						vel_msg.linear.y = 0.0
						vel_msg.linear.z = 0.0
						vel_msg.angular.x = 0.0
						vel_msg.angular.y = 0.0
						vel_msg.angular.z = 0.0
						self.velocity_publisher.publish(vel_msg)

						for elem in self.goals:
							self.marker_handler(x = elem['xcenter'], y = elem['ycenter'], z = None , mk_type = 0, action = 1, label = None, current = False, final = False)
				
						self.obj_id = None
						self.goals = []
						self.state_machine = 0

			
	def marker_handler(self,x,y,z,mk_type,action,label,current,final): #type 0 for navigation searching goals, 1 for found object, action 0 for adding, action 1 for removing
		Mk = Marker()
		Mk.header.frame_id = "map"
		Mk.id = 1
		Mk.type = Mk.SPHERE

		txt_marker = Marker()
		txt_marker.header.frame_id = "map"
		txt_marker.id = 1
		txt_marker.type = txt_marker.TEXT_VIEW_FACING


		goal_radius = 0.40
		obj_radius = 0.40
		
		if action == 0:
			#add markers
			Mk.action = Mk.ADD
			Mk.pose.position.x = x
			Mk.pose.position.y = y
						
			txt_marker.action = txt_marker.ADD
			txt_marker.pose.position.x = x 
			txt_marker.pose.position.y = y
			
			if mk_type == 0:
				
				Mk.pose.position.z = z
				Mk.ns = 'plcr_{:.2f}:{:.2f}'.format(x,y)
				Mk.scale.x = goal_radius
				Mk.scale.y = goal_radius
				Mk.scale.z = goal_radius
				if current == True:
					Mk.color.r = 0.99
					Mk.color.g = 0.99
					Mk.color.b = 0.180
					Mk.color.a = 0.9
				
				else:
					Mk.color.r = 0.917
					Mk.color.g = 0.180
					Mk.color.b = 0.180
					Mk.color.a = 0.8

				Mk.lifetime = rospy.Duration(0)

				txt_marker.pose.position.z = Mk.pose.position.z + goal_radius
				txt_marker.ns = 'plcr_txt_{:.2f}:{:.2f}'.format(x,y) #pose increment radius
				if current == True:
					txt_marker.text = 'Current Searching Goal'
				else:
					txt_marker.text = 'Searching Goal'

				txt_marker.color.r = 0.0
				txt_marker.color.g = 0.0
				txt_marker.color.b = 0.0
				txt_marker.lifetime = rospy.Duration(0)

			else:
				Mk.pose.position.z = z
				Mk.ns = 'objr_{:.2f}:{:.2f}'.format(x,y)
				Mk.scale.x = obj_radius
				Mk.scale.y = obj_radius
				Mk.scale.z = obj_radius
				Mk.color.r = 0.0
				Mk.color.g = 1.0
				Mk.color.b = 0.0
				Mk.color.a = 1.0
				Mk.lifetime = rospy.Duration(15)

				txt_marker.pose.position.z = Mk.pose.position.z + goal_radius
				txt_marker.ns = 'objr_txt_{:.2f}:{:.2f}'.format(x,y) #pose increment radius
				txt_marker.text = '{} Found!!!'.format(label)
				txt_marker.color.r = 0.0
				txt_marker.color.g = 1.0
				txt_marker.color.b = 0.0
				txt_marker.lifetime = rospy.Duration(15)


			Mk.pose.orientation.w = 1.0
			Mk.frame_locked = 1
			Mk.header.stamp = rospy.Time.now()
			if final == False:
				self.marker_pub.publish(Mk)
				
			rospy.sleep(0.03)
			
			txt_marker.color.a = 1
			txt_marker.pose.orientation.w = 1.0
			txt_marker.frame_locked = 1
			txt_marker.scale.z = 0.25
			txt_marker.color.a = 1
			txt_marker.header.stamp = rospy.Time.now()
			self.marker_pub.publish(txt_marker)
			rospy.sleep(0.03)


		if action == 1: #DELETE GOAL MARKER AND ITS TEXT MARKER
			Mk.action = Mk.DELETE
			Mk.ns = 'plcr_{:.2f}:{:.2f}'.format(x,y) 
			Mk.header.stamp = rospy.Time.now()
			self.marker_pub.publish(Mk)
			rospy.sleep(0.03)

			txt_marker = Marker()
			txt_marker.header.frame_id = "map"
			txt_marker.id = 1
			txt_marker.type = 9
			txt_marker.ns = 'plcr_txt_{:.2f}:{:.2f}'.format(x,y)
			txt_marker.action = txt_marker.DELETE 
			txt_marker.header.stamp = rospy.Time.now()
			self.marker_pub.publish(txt_marker)
			rospy.sleep(0.03)

		if action == 2:
			txt_marker.action = txt_marker.ADD
			txt_marker.pose.position.x = x 
			txt_marker.pose.position.y = y
			txt_marker.pose.position.z = 0.75
			txt_marker.ns = 'fail_{:.2f}:{:.2f}'.format(x,y) #pose increment radius
			txt_marker.text = 'Navigation Failed (Searching Aborted)'
			txt_marker.color.r = 1.0
			txt_marker.color.g = 0.0
			txt_marker.color.b = 0.0
			txt_marker.color.a = 1.0
			txt_marker.lifetime = rospy.Duration(8)
			txt_marker.pose.orientation.w = 1.0
			txt_marker.frame_locked = 1
			txt_marker.scale.z = 0.35
			txt_marker.header.stamp = rospy.Time.now()
			self.marker_pub.publish(txt_marker)
			rospy.sleep(0.03)


	def object_search(self, request):

		label = raw_input("Which object are you searching for? >> ")
		
		ret_dict = {'SConcept':{'used':False, 'id':None}, 'Concept':None , 'DirectPlaces':None, 'NearbyObjects':None, 'DerivedPlaces':None, 'success':None }
		go_search = False

		if label in self._SuperObjectConcepts:
			objects_related_idx =  np.argwhere( self._SuperObjObj[self._SuperObjectConcepts.index(label),:]).ravel()
			objs_related_labels = [ self._ObjConcepts[idx] for idx in objects_related_idx ]

			print 'From ', colored(label, 'cyan'), ' I know: ', colored(objs_related_labels,'green')

			entry = raw_input('Which one do you search for? >> ')
			while entry not in objs_related_labels:
				print colored('Wrong answer, this is not known by me!!','red')
				print 'From ', colored(label, 'cyan'), ' I know: ', colored(objs_related_labels,'green')

				entry = raw_input('Which one do you search for? >> ')


			ret_dict['Sconcept'] = {'used':True, 'id':self._SuperObjectConcepts.index(label)}
			label = entry

		if label in self._ObjConcepts:
			ret_dict['Concept'] = self._ObjConcepts.index(label)

			places_related = np.argwhere( self._ObjPlace[self._ObjConcepts.index(label), :] ).ravel()
			ret_dict['DirectPlaces'] = places_related

			if len(places_related) > 0:
				print colored(label,'cyan'), ' can be ', colored('AtLocation: ','blue'), colored([self._PlacesConcepts[place] for place in places_related],'green')

			objects_related_from_nearby = np.argwhere( self._ObjObj[self._ObjConcepts.index(label), :] ).ravel()

			if len(objects_related_from_nearby) > 0:
				print colored(label, 'cyan'), ' can be found ', colored('NearBy: ','blue'), colored([self._ObjConcepts[obj] for obj in objects_related_from_nearby], 'green')
			

			bl_pose = PoseStamped() 
			bl_pose.header.frame_id = "base_link" 
			bl_pose.pose.orientation.w = 1.0
			bl_pose.pose.position.x = 0.0
			bl_pose.pose.position.y = 0.0

			transform = self.tfBuffer.lookup_transform("map", "base_link", rospy.Time(0), rospy.Duration(1.0))
			
			bot_in_world = tf2_geometry_msgs.do_transform_pose(bl_pose, transform) #get current location of the robot in map frame 
			px = bot_in_world.pose.position.x
			py = bot_in_world.pose.position.y

			full_nav_list = []
			self.search_id = self._ObjConcepts.index(label)
			obj_locals = self.obj_in_map_srv(req_id=1, object_id = [self.search_id], radius = 0.0, xmin = 0.0, xmax = 0.0, ymin = 0.0, ymax = 0.0)
			for local in obj_locals.objects_in_map:
				aux_dist = np.power(np.power(local.pos_x-px, 2) + np.power(local.pos_y-py,2),0.5)
				aux_dict = {'xcenter':local.pos_x,'ycenter':local.pos_y,'zcenter':local.pos_z,'distance':aux_dist}
				full_nav_list.append(deepcopy(aux_dict))
				print colored('Added [x,y]=[{:.2f},{:.2f}] to nav_goals as there is a {} in the semantic map','green').format(local.pos_x,local.pos_y,self._ObjConcepts[self.search_id])

			print objects_related_from_nearby
			if len(objects_related_from_nearby) > 0:
				obj_locals = self.obj_in_map_srv(req_id=1, object_id = objects_related_from_nearby, radius = 0.0, xmin = 0.0, xmax = 0.0, ymin = 0.0, ymax = 0.0)
				for local in obj_locals.objects_in_map:
					add_nearby = True
					for navg in full_nav_list:
						if np.power(np.power(local.pos_x-navg['xcenter'], 2) + np.power(local.pos_y-navg['ycenter'],2),0.5) < 1.0:
							add_nearby = False
							
					if add_nearby == True:
						aux_dist = np.power(np.power(local.pos_x-px, 2) + np.power(local.pos_y-py,2),0.5)
						aux_dict = {'xcenter':local.pos_x,'ycenter':local.pos_y,'zcenter':local.pos_z,'distance':aux_dist}
						full_nav_list.append(deepcopy(aux_dict))
						print colored('Added [x,y]=[{:.2f},{:.2f}] to nav_goals as there is a {} in the semantic map and {} can be found NearBy it!!!','green').format(local.pos_x,local.pos_y, self._ObjConcepts[local.id], self._ObjConcepts[self.search_id])

			if len(places_related) > 0:
				plc_locals = self.plc_in_map_srv(self.min_scene_belief)
				for room in plc_locals.places: 
					direct_identique = set(places_related) & set(room.scene_ids)
					if len(direct_identique) > 0:
						x_plc_center = (room.xmax + room.xmin)/2
						y_plc_center = (room.ymax + room.ymin)/2
						aux_dist = np.power(np.power(x_plc_center-px, 2) + np.power(y_plc_center-py,2),0.5)
						aux_dict = {'xcenter':x_plc_center,'ycenter':y_plc_center,'zcenter':0.75,'distance':aux_dist}
						full_nav_list.append(deepcopy(aux_dict))
						print colored('Added [x,y]=[{:.2f},{:.2f}] to nav_goals as there is a high probability of finding a {} in that place (AtLocation edge)!!!','green').format(x_plc_center,y_plc_center, self._ObjConcepts[self.search_id])

			full_nav_list = sorted(full_nav_list,key=lambda k:k['distance'])
			self.goals = full_nav_list

			for j in xrange(0,len(self.goals)):
				if j == 0:
					ax = True
				else:
					ax = False

				self.marker_handler(x = self.goals[j]['xcenter'], y = self.goals[j]['ycenter'], z = self.goals[j]['zcenter'] , mk_type = 0, action = 0, label = None, current = ax, final = False)
	
			#print 'Goals_List:{}'.format(self.goals)

			print colored('Next Searching Goal -> x:{:.2f}, y{:.2f}','green').format(self.goals[0]['xcenter'],self.goals[0]['ycenter'])
		
			goal = MoveBaseGoal()
			goal.target_pose.header.frame_id = "map"
			goal.target_pose.header.stamp = rospy.Time.now()
			goal.target_pose.pose.position.x = self.goals[0]['xcenter']
			goal.target_pose.pose.position.y = self.goals[0]['ycenter']
			goal.target_pose.pose.orientation.w = 1.0
			self.move_base_client.send_goal(goal)

			self.state_machine = 1


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

	SuperObject_concepts = ['vehicles', 'animals', 'food', 'electronics']
	
	objects_in_map_service = rospy.get_param('~obj_service', default = '/object_mapper_node/obj_instances') 
	places_in_map_service = rospy.get_param('~places_service', default = '/place_mapper/place_cat_srv')
	obj_finder_srv_namespace = rospy.get_param('~obj_finder_srv_namespace', default = '~obj_finder_srv')
	joystick_buttons_topic = rospy.get_param('~joystick_buttons_topic', default = '/joy')

	SuperObjects_Objects_Relations = np.load(os.path.join(rospack.get_path('object_finder'),'files',rospy.get_param('~SupObjObj_file', default = 'SuperObjects_Objects_Relations.npy')))
	Object_LocatedAt_Place_Relations = np.load(os.path.join(rospack.get_path('object_finder'),'files',rospy.get_param('~ObjPlace_file', default = 'Object_LocatedAt_Place_Relations.npy')))
	Object_NearBy_Object_Relations =  np.load(os.path.join(rospack.get_path('object_finder'),'files',rospy.get_param('~ObjObj_file', default = 'Object_NearBy_Object_Relations.npy')))

	sub_detections_topic = rospy.get_param('~sub_detections', default = '/object_detection_dk/detected_objects')
	obj_searcher = ObjectSearcher(obj_in_map_srv = objects_in_map_service, plc_in_map_srv = places_in_map_service, obj_finder_srv_namespace = obj_finder_srv_namespace, 
									  detections_topic = sub_detections_topic, joystick_topic = joystick_buttons_topic, sub_pointcloud_topic = '/camera/depth_registered/points', ObjConcepts = Objects_concepts, PlacesConcepts = Places_concepts, SuperObjectConcepts = SuperObject_concepts, 
									  pub_markers_topic = '~reasoning_markers' , SuperObjObj = SuperObjects_Objects_Relations, ObjPlace = Object_LocatedAt_Place_Relations, ObjObj = Object_NearBy_Object_Relations)

		
	rospy.spin()
