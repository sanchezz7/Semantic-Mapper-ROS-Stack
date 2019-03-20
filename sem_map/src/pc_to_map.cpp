#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include "pcl_ros/transforms.h"
#include <tf/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <time.h>
#include <ctime>


class pc_transformer{

	public:
		pc_transformer(const ros::Publisher& pc_map_publisher): pointcloud_map_publisher(pc_map_publisher){
		}

		void pc_callback(const sensor_msgs::PointCloud2& msg){
      
			//clock_t shizzle = clock();		
      //time(&start);
      ros::Time t = ros::Time(0);
			try{
          			//listener.waitForTransform("map", "camera_depth_optical_frame", ros::Time::now(), ros::Duration(1.0));
  				listener.lookupTransform("map", "camera_depth_optical_frame", ros::Time(0), transform);
    		}
			catch (tf::TransformException ex){
      			ROS_ERROR("%s",ex.what());
    		}

    		bool transform_verf = pcl_ros::transformPointCloud("map", msg, map_cloud, listener);
    		if (!transform_verf){
    			ROS_WARN("PCL could not transform the cloud");
    		}
    		else{
    			pointcloud_map_publisher.publish(map_cloud);
    		}
        
		}

	private:
		tf::TransformListener listener; 
    		tf::StampedTransform transform;  	
    		ros::Publisher pointcloud_map_publisher;
    		sensor_msgs::PointCloud2 map_cloud;
};


int main(int argc, char** argv){
  	ros::init(argc, argv, "pcl");
  	ros::NodeHandle nh;
  	ros::Publisher pointcloud_map_publisher = nh.advertise<sensor_msgs::PointCloud2>("pcl/map_cloud",1, true);
  	pc_transformer transformer(pointcloud_map_publisher);
  	ros::Subscriber point_cloud_sub = nh.subscribe("/camera/depth_registered/points",1,&pc_transformer::pc_callback,&transformer);
  	

  	ros::spin();
  	return 0;
}
