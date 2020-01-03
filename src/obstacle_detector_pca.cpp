#include "obstacle_detector_pca/obstacle_detector_pca.h"

ObstacleDetectorPCA::ObstacleDetectorPCA(void)
:local_nh("~")
{
    downsampled_cloud_pub = local_nh.advertise<sensor_msgs::PointCloud2>("downsampled_cloud", 1);
    clustered_cloud_pub = local_nh.advertise<sensor_msgs::PointCloud2>("clustered_cloud", 1);
    bb_pub = local_nh.advertise<visualization_msgs::MarkerArray>("bounding_boxes", 1);
    cloud_sub = nh.subscribe("/velodyne_obstacles", 1, &ObstacleDetectorPCA::cloud_callback, this, ros::TransportHints().reliable().tcpNoDelay(true));

    local_nh.param<double>("LEAF_SIZE", LEAF_SIZE, {0.1});
    local_nh.param<double>("TOLERANCE", TOLERANCE, {0.3});
    local_nh.param<int>("MIN_CLUSTER_SIZE", MIN_CLUSTER_SIZE, {30});
    local_nh.param<int>("MAX_CLUSTER_SIZE", MAX_CLUSTER_SIZE, {2000});
    local_nh.param<double>("MIN_HEIGHT", MIN_HEIGHT, {1.5});
    local_nh.param<double>("MAX_HEIGHT", MAX_HEIGHT, {1.9});
    local_nh.param<double>("MIN_WIDTH", MIN_WIDTH, {0.4});
    local_nh.param<double>("MAX_WIDTH", MAX_WIDTH, {1.0});

    cloud_ptr = CloudXYZINPtr(new CloudXYZIN);

    std::cout << "LEAF_SIZE: "<< LEAF_SIZE << std::endl;
    std::cout << "TOLERANCE: "<< TOLERANCE << std::endl;
    std::cout << "MIN_CLUSTER_SIZE: "<< MIN_CLUSTER_SIZE << std::endl;
    std::cout << "MAX_CLUSTER_SIZE: "<< MAX_CLUSTER_SIZE << std::endl;
    std::cout << "MIN_HEIGHT: "<< MIN_HEIGHT << std::endl;
    std::cout << "MAX_HEIGHT: "<< MAX_HEIGHT << std::endl;
    std::cout << "MIN_WIDTH: "<< MIN_WIDTH << std::endl;
    std::cout << "MAX_WIDTH: "<< MAX_WIDTH << std::endl;
}

void ObstacleDetectorPCA::cloud_callback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    std::cout << "=== obstacle_detector_pca ===" << std::endl;

    double start_time = ros::Time::now().toSec();

    pcl::fromROSMsg(*msg, *cloud_ptr);
    std::cout << "subscribed cloud size: " << cloud_ptr->points.size() << std::endl;

    pcl::VoxelGrid<PointXYZIN> vg;
    vg.setInputCloud(cloud_ptr);
    vg.setLeafSize(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE);
    vg.filter(*cloud_ptr);
    std::cout << "downsampled cloud size: " << cloud_ptr->points.size() << std::endl;
    downsampled_cloud_pub.publish(*cloud_ptr);

    std::vector<pcl::PointIndices> cluster_indices;
    get_euclidean_cluster_indices(cluster_indices);
    std::cout << "time: " << ros::Time::now().toSec() - start_time << "[s]" << std::endl;

    std::vector<CloudXYZINPtr> clusters;
    get_euclidean_clusters(cluster_indices, clusters);
    std::cout << "time: " << ros::Time::now().toSec() - start_time << "[s]" << std::endl;

    visualization_msgs::MarkerArray bbs;
    int cluster_num = clusters.size();
    std::cout << "cluster num: " << cluster_num << std::endl;
    static int last_num_of_bbs = 0;
    std::cout << "last bb num: " << last_num_of_bbs << std::endl;
    int bbs_num = 0;
    for(int i=0;i<cluster_num;i++){
        // std::cout << i << std::endl;;
        double yaw = 0;
        Eigen::Vector3d centroid;
        Eigen::Vector3d scale;
        principal_component_analysis(clusters[i], yaw, centroid, scale);
        if(MIN_HEIGHT < scale(2) && scale(2) < MAX_HEIGHT
           && MIN_WIDTH < scale(0) && scale(0) < MAX_WIDTH
           && MIN_WIDTH < scale(1) && scale(1) < MAX_WIDTH
        ){
            bounding_box_lib::BoundingBox bb;
            bb.set_id(bbs_num);
            bb.set_frame_id(msg->header.frame_id);
            bb.set_orientation(0, 0, yaw);
            bb.set_scale(scale(0), scale(1), scale(2));
            bb.set_centroid(centroid(0), centroid(1), centroid(2));
            bb.set_rgb(0, 200, 255);
            bb.calculate_vertices();
            bbs.markers.push_back(bb.get_bounding_box());
            std::cout << "id: " << bbs_num << std::endl;
            std::cout << "centroid: " << centroid.transpose() << std::endl;
            std::cout << "scale: " << scale.transpose() << std::endl;
            bbs_num++;
        }else{
            std::cout << "rejected" << std::endl;
            std::cout << "centroid: " << centroid.transpose() << std::endl;
            std::cout << "scale: " << scale.transpose() << std::endl;
        }
    }
    std::cout << "bbs num: " << bbs_num << std::endl;
    for(int i=bbs_num;i<last_num_of_bbs;i++){
        visualization_msgs::Marker m;
        m.action = visualization_msgs::Marker::DELETE;
        m.id = i;
        m.pose.orientation = tf::createQuaternionMsgFromYaw(0);
        bbs.markers.push_back(m);
    }
    bb_pub.publish(bbs);
    last_num_of_bbs = bbs_num;

    std::cout << "time: " << ros::Time::now().toSec() - start_time << "[s]" << std::endl;
}

void ObstacleDetectorPCA::get_euclidean_cluster_indices(std::vector<pcl::PointIndices>& cluster_indices)
{
    pcl::search::KdTree<PointXYZIN>::Ptr tree(new pcl::search::KdTree<PointXYZIN>);
    tree->setInputCloud(cloud_ptr);
    pcl::EuclideanClusterExtraction<PointXYZIN> ec;
    ec.setClusterTolerance(TOLERANCE);
    ec.setMinClusterSize(MIN_CLUSTER_SIZE);
    ec.setMaxClusterSize(MAX_CLUSTER_SIZE);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_ptr);
    ec.extract(cluster_indices);
    std::cout << cluster_indices.size() << " clusters are extracted" << std::endl;
}

void ObstacleDetectorPCA::get_euclidean_clusters(const std::vector<pcl::PointIndices>& cluster_indices, std::vector<CloudXYZINPtr>& clusters)
{
    int cluster_num = cluster_indices.size();
    CloudRGBPtr colored_clusters(new CloudRGB);
    colored_clusters->header = cloud_ptr->header;
    colored_clusters->points.reserve(cloud_ptr->points.size());
    int cluster_count = 0;
    for(auto it=cluster_indices.begin();it!=cluster_indices.end();++it){
        CloudXYZINPtr cluster(new CloudXYZIN);
        cluster->points.reserve(it->indices.size());
        for(auto pit=it->indices.begin();pit!=it->indices.end();++pit){
            cluster->points.push_back(cloud_ptr->points[*pit]);
            pcl::PointXYZHSV p_hsv;
            PointRGB p_rgb;
            p_hsv.x = cloud_ptr->points[*pit].x;
            p_hsv.y = cloud_ptr->points[*pit].y;
            p_hsv.z = cloud_ptr->points[*pit].z;
            p_hsv.h = 255.0 * cluster_count / (double)cluster_num;
            p_hsv.s = 1.0;
            p_hsv.v = 1.0;
            pcl::PointXYZHSVtoXYZRGB(p_hsv, p_rgb);
            colored_clusters->points.push_back(p_rgb);
        }
        // std::cout << cluster_count << ": " << 255.0 * cluster_count / (double)cluster_num << std::endl;;
        clusters.push_back(cluster);
        cluster_count++;
    }
    clustered_cloud_pub.publish(colored_clusters);
}

void ObstacleDetectorPCA::principal_component_analysis(const CloudXYZINPtr& cluster, double& yaw, Eigen::Vector3d& centroid, Eigen::Vector3d& scale)
{
    double cluster_size = cluster->points.size();
    if(cluster_size < 3){
        return;
    }
    double ave_x = 0;
    double ave_y = 0;
    double ave_z = 0;
    for(auto& pt : cluster->points){
        ave_x += pt.x;
        ave_y += pt.y;
        ave_z += pt.z;
    }
    ave_x /= cluster_size;
    ave_y /= cluster_size;
    ave_z /= cluster_size;
    centroid << ave_x, ave_y, ave_z;
    double sigma_xx = 0;
    double sigma_xy = 0;
    double sigma_yy = 0;
    Eigen::Vector3d max_xyz(cluster->points[0].x, cluster->points[0].y, cluster->points[0].z);;
    Eigen::Vector3d min_xyz(cluster->points[0].x, cluster->points[0].y, cluster->points[0].z);;
    for(auto& pt : cluster->points){
        sigma_xx += (pt.x - ave_x) * (pt.x - ave_x);
        sigma_xy += (pt.x - ave_x) * (pt.y - ave_y);
        sigma_yy += (pt.y - ave_y) * (pt.y - ave_y);
        max_xyz(0) = std::max(max_xyz(0), (double)pt.x);
        max_xyz(1) = std::max(max_xyz(1), (double)pt.y);
        max_xyz(2) = std::max(max_xyz(2), (double)pt.z);
        min_xyz(0) = std::min(min_xyz(0), (double)pt.x);
        min_xyz(1) = std::min(min_xyz(1), (double)pt.y);
        min_xyz(2) = std::min(min_xyz(2), (double)pt.z);
    }
    sigma_xx /= cluster_size;
    sigma_xy /= cluster_size;
    sigma_yy /= cluster_size;
    scale = max_xyz - min_xyz;
    Eigen::Matrix2d cov_mat;
    cov_mat << sigma_xx, sigma_xy,
               sigma_xy, sigma_yy;
    Eigen::EigenSolver<Eigen::Matrix2d> es(cov_mat);
    Eigen::Vector2d eigen_values = es.eigenvalues().real();
    Eigen::Matrix2d eigen_vectors = es.eigenvectors().real();
    // std::cout << ave_x << ", " << ave_y << std::endl;
    // std::cout << cov_mat << std::endl;
    // std::cout << eigen_values << std::endl;
    // std::cout << eigen_vectors << std::endl;
    double length = max_xyz(0) - min_xyz(0);
    double width = max_xyz(1) - min_xyz(1);
    double height = max_xyz(2) - min_xyz(2);
    // std::cout << length << ", " << width << ", " << height << std::endl;
    int larger_index = 0;
    if(eigen_values(0) > eigen_values(1)){
        larger_index = 0;
    }else{
        larger_index = 1;
    }
    Eigen::Vector2d larger_vector = eigen_vectors.col(larger_index);
    yaw = atan2(larger_vector(1), larger_vector(0));
    // std::cout << yaw << "[rad]" << std::endl;
}

void ObstacleDetectorPCA::process(void)
{
    ros::spin();
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "obstacle_detector_pca");
    ObstacleDetectorPCA obstacle_detector_pca;
    obstacle_detector_pca.process();
    return 0;
}
