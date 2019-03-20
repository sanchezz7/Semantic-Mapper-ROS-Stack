import numpy as np 
from nltk.tokenize import sent_tokenize, word_tokenize
import os
from termcolor import colored
import time

class Ontologie(object):
	
	def __init__(self, ObjConcepts, PlacesConcepts, SuperObjectConcepts, SuperObjObj, ObjPlace, ObjObj):
		
		self._ObjConcepts = ObjConcepts #list of object labels
		self._PlacesConcepts = PlacesConcepts #list of places labels
		self._SuperObjectConcepts = SuperObjectConcepts #list of superconcepts labels [vehicles, animals, food, electronics]
		self._SuperObjObj = SuperObjObj #super objects to objects relations (e.g. apple is a food)
										#boolean np.array dim(4,80) (superconcept, object) describing if there is relation or not between the object and the super category
		self._ObjPlace = ObjPlace # boolean np.array describing the relations between objects and places (80,205) (AtLocationRelation)
		self._ObjObj = ObjObj #boolena numpy array describing NearbyRelation (80,80) apple Nearby Banana = TRue, therefore self._ObjObj[apple_idx, banana_idx] = True

	def object_search(self, label):
		

		ret_dict = {'SConcept':{'used':False, 'id':None}, 'Concept':None , 'DirectPlaces':None, 'NearbyObjects':None, 'DerivedPlaces':None, 'success':None }

		if label in self._SuperObjectConcepts:
			objects_related_idx =  np.argwhere( self._SuperObjObj[self._SuperObjectConcepts.index(label),:]).ravel()
			#print objects_related_idx
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
				plcs = []
				for obj_related in objects_related_from_nearby:
					nb_places_related = np.argwhere(self._ObjPlace[obj_related, :]).ravel()
					
					if len(nb_places_related) > 0:
						print colored(self._ObjConcepts[obj_related],'magenta'), colored(' AtLocation: ','blue'), colored([self._PlacesConcepts[plc] for plc in nb_places_related],'green')
						for elem in nb_places_related:
							plcs.append(elem)
						places_set = set()
						places_add = places_set.add
						plcs = [x for x in plcs if not (x in places_set or places_add(x))]
						print 'Therefore ', colored(label,'cyan'), 'might also be ', colored('AtLocation: ','blue'), colored([self._PlacesConcepts[plc] for plc in plcs], 'green')

			if ( (len(objects_related_from_nearby) == 0) and (len(places_related) == 0) ):
				print colored( 'I do not where such object could be located', 'red')

		else:
			print colored( 'I do not know which concept you are talking about!! Call this service again with a proper object query....','red')



		'''			
		elif label in self._ObjConcepts:
			places_related = np.argwhere(self._ObjPlace[self._ObjConcepts.index(label),:]).ravel()
			if len(places_related) > 0:
				print colored(label, 'cyan'), ' might be ', colored('AtLocation :','blue'), colored([self._PlacesConcepts[plc] for plc in places_related],'green')
			else:
				objects_related_from_nearby = np.argwhere( self._ObjObj[self._ObjConcepts.index(label), :] ).ravel()
				if len(objects_related_from_nearby) > 0:
					print colored(label, 'cyan'), ' can be found ', colored('NearBy: ','blue'), colored([self._ObjConcepts[obj] for obj in objects_related_from_nearby], 'green')
					plcs = []
					for obj_related in objects_related_from_nearby:
						places_related = np.argwhere(self._ObjPlace[obj_related, :]).ravel()
						if len(places_related) > 0:
							print colored(self._ObjConcepts[obj_related],'magenta'), colored(' AtLocation: ','blue'), colored([self._PlacesConcepts[plc] for plc in places_related],'green')
							for elem in places_related:
								plcs.append(elem)
					places_set = set()
					places_add = places_set.add
					plcs = [x for x in plcs if not (x in places_set or places_add(x))]
					print 'Therefore ', colored(label,'cyan'), 'might also be ', colored('AtLocation: ','blue'), colored([self._PlacesConcepts[plc] for plc in plcs], 'green')
				else:
					print colored('I do not know where {} might be located','red').format()

		else:
			print colored( 'I do not know which concept you are talking about!! Call this service again with a proper object query....','red')
		'''








		
if __name__ == '__main__':


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

	load = False

	if load == True:
		SuperObjects_Objects_Relations = np.load('SuperObjects_Objects_Relations.npy')
		Object_LocatedAt_Place_Relations = np.load('Object_LocatedAt_Place_Relations.npy')
		Object_NearBy_Object_Relations = np.load('Object_NearBy_Object_Relations.npy')

		ont = Ontologie(ObjConcepts = Objects_concepts , PlacesConcepts = Places_concepts , SuperObjectConcepts = SuperObject_concepts , 
						SuperObjObj = SuperObjects_Objects_Relations, ObjPlace = Object_LocatedAt_Place_Relations, ObjObj = Object_NearBy_Object_Relations)

		entry = raw_input('what do you want to search for? >> ')
		ont.object_search(entry)


	else:

		'''

		 m    m   mmm         mmmmm  mm   m mmmmm mmmmmmm
		 #  m"  m"   "          #    #"m  #   #      #   
		 #m#    #   mm          #    # #m #   #      #   
		 #  #m  #    #          #    #  # #   #      #   
		 #   "m  "mmm"        mm#mm  #   ## mm#mm    #   
														 
					  """"""                             

		'''
		'''
		SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) #current dir
		labels_file = 'categoryIndex_places205.csv'
		paths = list()
		with open(labels_file) as class_file:
			for j,line in enumerate (class_file):
				paths.append(line.strip().split(' ')[0][3:])
				slash_number = len(paths[j].split('/'))
				if slash_number == 2:
					aux = paths[j].split('/')
					paths[j] = aux[0]+'_'+aux[1] 
		print paths

		objects_names_file = 'coco.names'
		objects_concepts = []
		with open(objects_names_file) as fid:
			for obj_name in fid:
				objects_concepts.append(obj_name.strip('\n'))
		'''

		'''
		for obj in ['bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat']: 
			SuperObjects_objects_relations[0].append( Objects_concepts.index(obj) )

		for obj in ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']:
			SuperObjects_objects_relations[1].append(Objects_concepts.index(obj))

		for obj in ['banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake']:
			SuperObjects_objects_relations[2].append(Objects_concepts.index(obj))

		for obj in [ 'tv/monitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'refrigerator', 'clock']:
			SuperObjects_objects_relations[3].append(Objects_concepts.index(obj))
		'''

		SuperObjects_Objects_Relations = np.zeros( (4,80), dtype = np.bool)

		sup_objects = [ [1, 2, 3, 4, 5, 6, 7, 8], 
						[14, 15, 16, 17, 18, 19, 20, 21, 22, 23], 
						[46, 47, 48, 49, 50, 51, 52, 53, 54, 55], 
						[62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 74] ]
		
		for j in xrange(0,4):
			SuperObjects_Objects_Relations[j, sup_objects[j]] = True

		np.save('SuperObjects_Objects_Relations.npy', SuperObjects_Objects_Relations)

										  
		'''
		SuperObject_LocatedAt_relations = [ ['bridge', 'forest_road', 'highway', 'gas_station', 'driveway', 'parking_lot'], 

												'forest_path', forest_road, living room, mountain, ocean, swamp, veranda, yard

												kitchen, kitchennete, dinning room, living room, food_court, cafeteria, restaurant, restaurant_kitchen, dinnete_home, coffe shop, bar , bakery shop, picnic_area, supermarket

												office home office livingroom kitchen ]  
		'''


		Object_LocatedAt_Place_Relations = np.zeros((80,205), dtype = np.bool)
		print Object_LocatedAt_Place_Relations.shape
		print Object_LocatedAt_Place_Relations.ndim
		#adding cup, fork, knife, spoon, bowl to (kitchen, kitchennete, dining room, dinnete (wine glass also added to living room, bar)
		places_idx = [ Places_concepts.index(place) for place in ['kitchen', 'kitchenette', 'dining_room', 'dinette_home' ] ]
		for obj in ['cup', 'fork', 'knife', 'spoon', 'bowl']:
			obj_idx = Objects_concepts.index(obj)
			Object_LocatedAt_Place_Relations[obj_idx, places_idx] = True


		#adding wine_glass to kitchen, kitchennete, dining room, dinnete living room, bar
		places_idx =  [ Places_concepts.index(place) for place in ['kitchen', 'kitchenette', 'dining_room', 'dinette_home', 'living_room', 'bar' ] ]
		obj_idx = Objects_concepts.index('wine glass')
		Object_LocatedAt_Place_Relations[obj_idx, places_idx] = True

		#adding banana sandwich orange brocolli carrot hot dog pizza donut cake to (kitchen, kitchennete, dining_room, dinnete_home, pantry, cafetaria, bar)
		places_idx = [Places_concepts.index(place) for place in ['kitchen', 'kitchenette', 'dining_room', 'dinette_home', 'bar', 'cafeteria', 'coffee_shop', 'restaurant_kitchen', 'supermarket']]
		for obj in ['banana', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake']:
			obj_idx = Objects_concepts.index(obj)
			Object_LocatedAt_Place_Relations[obj_idx, places_idx] = True

		#adding chair and dining table to office, home office, living room, dinnete, dinnete_home, dining_room, bedroom, attic, classroom, lobby, parlor, veranda and waiting room
		places_idx = [ Places_concepts.index(place) for place in ['office', 'home_office', 'living_room', 'dinette_home', 'dining_room','bedroom', 'attic', 'classroom', 'lobby', 'parlor', 'veranda', 'waiting_room']]
		for obj in ['chair', 'diningtable']:
			obj_idx = Objects_concepts.index(obj)
			Object_LocatedAt_Place_Relations[obj_idx, places_idx] = True

		#adding toilet to shower
		places_idx = Places_concepts.index('shower')
		obj_idx = Objects_concepts.index('toilet')
		Object_LocatedAt_Place_Relations[obj_idx, places_idx] = True

		#adding bed to bedroom and hotel_room
		places_idx = [Places_concepts.index(place) for place in ['bedroom', 'hotel_room'] ]
		obj_idx = Objects_concepts.index('bed')
		Object_LocatedAt_Place_Relations[obj_idx, places_idx] = True

		#addind microwave, oven, toaster, sink, and refrigerator to kitchen and kitchenette
		places_idx = [Places_concepts.index(place) for place in ['kitchen', 'kitchenette']]
		for obj in ['microwave', 'oven', 'toaster', 'sink', 'refrigerator']:
			obj_idx = Objects_concepts.index(obj)
			Object_LocatedAt_Place_Relations[obj_idx, places_idx] = True

		#adding sofa places
		places_idx = [Places_concepts.index(place) for place in ['living_room', 'lobby', 'waiting_room', 'cafeteria', 'coffee_shop', 'bar']]
		obj_idx = Objects_concepts.index('sofa')
		Object_LocatedAt_Place_Relations[obj_idx, places_idx] = True


		#adding book places
		places_idx = [Places_concepts.index(place) for place in ['living_room', 'waiting_room', 'office', 'home_office', 'bedroom', 'classroom', 'auditorium']]
		obj_idx = Objects_concepts.index('book')
		Object_LocatedAt_Place_Relations[obj_idx, places_idx] = True


		#adding tv/monitor and remote
		places_idx = [Places_concepts.index(place) for place in ['living_room', 'waiting_room', 'office', 'home_office', 'classroom', 'auditorium', 'bedroom', 'bar', 'cafeteria']]
		for obj in ['tv/monitor', 'remote']:
			obj_idx = Objects_concepts.index(obj)
			Object_LocatedAt_Place_Relations[obj_idx, places_idx] = True


		#adding laptop, mouse, keyboard
		places_idx = [Places_concepts.index(place) for place in ['living_room', 'office', 'home_office', 'classroom', 'auditorium', 'bedroom']]
		for obj in ['laptop', 'mouse', 'keyboard']:
			obj_idx = Objects_concepts.index(obj)
			Object_LocatedAt_Place_Relations[obj_idx, places_idx] = True

		np.save('Object_LocatedAt_Place_Relations.npy', Object_LocatedAt_Place_Relations)



		Object_NearBy_Object_Relations = np.zeros((80,80), dtype = np.bool)

		#adding apple nearby banana
		obj_idx = Objects_concepts.index('apple')
		near_obj_idx = [Objects_concepts.index(obj) for obj in ['banana','orange']]
		Object_NearBy_Object_Relations[obj_idx, near_obj_idx] = True

		#adding chair nearby sofa and dinningtable
		obj_idx = Objects_concepts.index('chair')
		near_obj_idx = [Objects_concepts.index(obj_label) for obj_label in ['sofa', 'diningtable']]
		Object_NearBy_Object_Relations[obj_idx, near_obj_idx] = True

		#adding mouse, keyboard, tv/monitor nearby laptop
		obj_idx = [Objects_concepts.index(obj_label) for obj_label in ['mouse', 'keyboard', 'tv/monitor']]
		near_obj_idx = Objects_concepts.index('laptop')
		Object_NearBy_Object_Relations[obj_idx, near_obj_idx] = True

		np.save('Object_NearBy_Object_Relations.npy', Object_NearBy_Object_Relations)
		
		ont = Ontologie(ObjConcepts = Objects_concepts , PlacesConcepts = Places_concepts , SuperObjectConcepts = SuperObject_concepts , 
						SuperObjObj = SuperObjects_Objects_Relations, ObjPlace = Object_LocatedAt_Place_Relations, ObjObj = Object_NearBy_Object_Relations)

		entry = raw_input('what do you want to search for? >> ')
		ont.object_search(entry)

