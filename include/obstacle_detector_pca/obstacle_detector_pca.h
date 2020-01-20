#ifndef __OBSTACLE_DETECTOR_PCA_H
#define __OBSTACLE_DETECTOR_PCA_H

#include <ros/ros.h>
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_types_conversion.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <Eigen/Dense>

#include "bounding_box_lib/bounding_box.h"

class ObstacleDetectorPCA
{
public:
    typedef pcl::PointXYZINormal PointXYZIN;
    typedef pcl::PointCloud<PointXYZIN> CloudXYZIN;
    typedef pcl::PointCloud<PointXYZIN>::Ptr CloudXYZINPtr;
    typedef pcl::PointXYZRGB PointRGB;
    typedef pcl::PointCloud<PointRGB> CloudRGB;
    typedef pcl::PointCloud<PointRGB>::Ptr CloudRGBPtr;

    ObstacleDetectorPCA(void);

    void cloud_callback(const sensor_msgs::PointCloud2ConstPtr&);
    void get_euclidean_cluster_indices(std::vector<pcl::PointIndices>&);
    void get_euclidean_clusters(const std::vector<pcl::PointIndices>&, std::vector<CloudXYZINPtr>&);
    void principal_component_analysis(const CloudXYZINPtr& cluster, double& yaw, Eigen::Vector3d&, Eigen::Vector3d&);
    bool is_human_cluster(const Eigen::Vector3d&, const Eigen::Vector3d&);
    void process(void);

private:
    double LEAF_SIZE;
    double TOLERANCE;
    int MAX_CLUSTER_SIZE;
    int MIN_CLUSTER_SIZE;
    double MIN_HEIGHT;
    double MAX_HEIGHT;
    double MIN_WIDTH;
    double MAX_WIDTH;
    double MIN_LENGTH;
    double MAX_LENGTH;
    double LIDAR_HEIGHT_FROM_GROUND;
    double LIDAR_VERTICAL_FOV_UPPER;
    double LIDAR_VERTICAL_FOV_LOWER;
    double LIDAR_LINES;
    double LIDAR_ANGLE;

    ros::NodeHandle nh;
    ros::NodeHandle local_nh;

    ros::Publisher downsampled_cloud_pub;
    ros::Publisher clustered_cloud_pub;
    ros::Publisher bb_pub;
    ros::Publisher obstacle_removed_cloud_pub;
    ros::Publisher obstacle_pose_pub;
    ros::Subscriber cloud_sub;

    CloudXYZINPtr cloud_ptr;
};

#endif// __OBSTACLE_DETECTOR_PCA_H
