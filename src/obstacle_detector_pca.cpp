#include "obstacle_detector_pca/obstacle_detector_pca.h"

ObstacleDetectorPCA::ObstacleDetectorPCA(void)
:local_nh("~")
{
    downsampled_cloud_pub = local_nh.advertise<sensor_msgs::PointCloud2>("downsampled_cloud", 1);
    clustered_cloud_pub = local_nh.advertise<sensor_msgs::PointCloud2>("clustered_cloud", 1);
    bb_pub = local_nh.advertise<visualization_msgs::MarkerArray>("bounding_boxes", 1);
    obstacle_removed_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/cloud/obstacle_removed", 1);
    obstacle_pose_pub = nh.advertise<geometry_msgs::PoseArray>("/dynamic_obstacles", 1);
    cloud_sub = nh.subscribe("/velodyne_obstacles", 1, &ObstacleDetectorPCA::cloud_callback, this, ros::TransportHints().reliable().tcpNoDelay(true));

    local_nh.param<double>("LEAF_SIZE", LEAF_SIZE, {0.1});
    local_nh.param<double>("TOLERANCE", TOLERANCE, {0.30});
    local_nh.param<int>("MIN_CLUSTER_SIZE", MIN_CLUSTER_SIZE, {10});
    local_nh.param<int>("MAX_CLUSTER_SIZE", MAX_CLUSTER_SIZE, {2000});
    local_nh.param<double>("MIN_HEIGHT", MIN_HEIGHT, {1.1});
    local_nh.param<double>("MAX_HEIGHT", MAX_HEIGHT, {1.9});
    local_nh.param<double>("MIN_WIDTH", MIN_WIDTH, {0.4});
    local_nh.param<double>("MAX_WIDTH", MAX_WIDTH, {1.0});
    local_nh.param<double>("MIN_LENGTH", MIN_LENGTH, {0.2});
    local_nh.param<double>("MAX_LENGTH", MAX_LENGTH, {1.0});
    local_nh.param<double>("LIDAR_HEIGHT_FROM_GROUND", LIDAR_HEIGHT_FROM_GROUND, {1.2});
    local_nh.param<double>("LIDAR_VERTICAL_FOV_UPPER", LIDAR_VERTICAL_FOV_UPPER, {10.67 * M_PI / 180.0});
    local_nh.param<double>("LIDAR_VERTICAL_FOV_LOWER", LIDAR_VERTICAL_FOV_LOWER, {-30.67 * M_PI / 180.0});
    local_nh.param<double>("LIDAR_LINES", LIDAR_LINES, {32});

    LIDAR_ANGLE = (LIDAR_VERTICAL_FOV_UPPER - LIDAR_VERTICAL_FOV_LOWER) / LIDAR_LINES;

    cloud_ptr = CloudXYZINPtr(new CloudXYZIN);

    std::cout << "LEAF_SIZE: "<< LEAF_SIZE << std::endl;
    std::cout << "TOLERANCE: "<< TOLERANCE << std::endl;
    std::cout << "MIN_CLUSTER_SIZE: "<< MIN_CLUSTER_SIZE << std::endl;
    std::cout << "MAX_CLUSTER_SIZE: "<< MAX_CLUSTER_SIZE << std::endl;
    std::cout << "MIN_HEIGHT: "<< MIN_HEIGHT << std::endl;
    std::cout << "MAX_HEIGHT: "<< MAX_HEIGHT << std::endl;
    std::cout << "MIN_WIDTH: "<< MIN_WIDTH << std::endl;
    std::cout << "MAX_WIDTH: "<< MAX_WIDTH << std::endl;
    std::cout << "MIN_LENGTH: "<< MIN_LENGTH << std::endl;
    std::cout << "MAX_LENGTH: "<< MAX_LENGTH << std::endl;
    std::cout << "LIDAR_HEIGHT_FROM_GROUND: "<< LIDAR_HEIGHT_FROM_GROUND << std::endl;
    std::cout << "LIDAR_VERTICAL_FOV_UPPER: "<< LIDAR_VERTICAL_FOV_UPPER << std::endl;
    std::cout << "LIDAR_VERTICAL_FOV_LOWER: "<< LIDAR_VERTICAL_FOV_LOWER << std::endl;
    std::cout << "LIDAR_LINES: "<< LIDAR_LINES << std::endl;
}

void ObstacleDetectorPCA::cloud_callback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    std::cout << "=== obstacle_detector_pca ===" << std::endl;

    double start_time = ros::Time::now().toSec();

    geometry_msgs::PoseArray obstacle_poses;
    obstacle_poses.header = msg->header;

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

    pcl::PointIndices::Ptr remove_indices(new pcl::PointIndices());

    visualization_msgs::MarkerArray bbs;
    int cluster_num = clusters.size();
    std::cout << "cluster num: " << cluster_num << std::endl;
    static int last_num_of_bbs = 0;
    std::cout << "last bb num: " << last_num_of_bbs << std::endl;
    int bbs_num = 0;
    const double INNERMOST_RING_RADIUS = LIDAR_HEIGHT_FROM_GROUND / tan(fabs(LIDAR_VERTICAL_FOV_LOWER));
    std::cout << "INNERMOST_RING_RADIUS: " << INNERMOST_RING_RADIUS << std::endl;
    for(int i=0;i<cluster_num;i++){
        std::cout << i << std::endl;;
        double yaw = 0;
        Eigen::Vector3d centroid;
        Eigen::Vector3d scale;
        principal_component_analysis(clusters[i], yaw, centroid, scale);

        std::cout << "centroid: " << centroid.transpose() << std::endl;
        std::cout << "scale: " << scale.transpose() << std::endl;
        std::cout << "cluster size: " << clusters[i]->points.size() << std::endl;;

        double distance = centroid.segment(0, 2).norm();
        std::cout << "distance: " << distance << std::endl;
        if(distance < INNERMOST_RING_RADIUS){
            std::cout << "the obstacle is too close" << std::endl;
            double height_from_distance = distance * (tan(LIDAR_VERTICAL_FOV_UPPER) + tan(fabs(LIDAR_VERTICAL_FOV_LOWER)));
            std::cout << "height_from_distance: " << height_from_distance << std::endl;
            std::cout << "height: " << scale(2) << std::endl;
            if(fabs(scale(2) - height_from_distance) < 0.05){
                std::cout << "filled with obstacle" << std::endl;
                scale(2) = (MAX_HEIGHT + MIN_HEIGHT) * 0.5;
                centroid(2) = scale(2) * 0.5 - LIDAR_HEIGHT_FROM_GROUND;
            }
        }
        double lowest = std::max(LIDAR_HEIGHT_FROM_GROUND + centroid(2) - scale(2) * 0.5, 0.0);
        std::cout << "lowest: " << lowest << std::endl;
        double height_error_range = distance * sin(LIDAR_ANGLE);
        std::cout << "height_error_range: " << height_error_range << std::endl;
        if(0 < lowest && lowest < height_error_range){
            std::cout << "lost under part of the obstacle?" << std::endl;
            centroid(2) -= lowest;
            scale(2) += lowest;
            std::cout << "centroid: " << centroid.transpose() << std::endl;
            std::cout << "scale: " << scale.transpose() << std::endl;
        }

        std::cout << "height : " << scale(2) << std::endl;
        if(scale(2) + height_error_range < MAX_HEIGHT){
            scale(2) += height_error_range;
            std::cout << "height was modified: " << scale(2) << std::endl;
        }
        if(!(MIN_HEIGHT < scale(2) && scale(2) < MAX_HEIGHT)){
            std::cout << "invalid height" << std::endl;
            continue;
        }
        if(is_human_cluster(centroid, scale)){
            std::copy(cluster_indices[i].indices.begin(), cluster_indices[i].indices.end(), std::back_inserter(remove_indices->indices));
            bounding_box_lib::BoundingBox bb;
            bb.set_id(bbs_num);
            bb.set_frame_id(msg->header.frame_id);
            bb.set_orientation(0, 0, yaw);
            bb.set_scale(scale(0), scale(1), scale(2));
            bb.set_centroid(centroid(0), centroid(1), centroid(2));
            bb.set_rgb(0, 200, 255);
            bb.calculate_vertices();
            bbs.markers.push_back(bb.get_bounding_box());
            std::cout << "accepted" << std::endl;
            std::cout << "\033[32mid: " << bbs_num << "\033[0m" << std::endl;
            geometry_msgs::Pose p;
            p.position.x = centroid(0);
            p.position.y = centroid(1);
            p.position.z = centroid(2);
            p.orientation = tf::createQuaternionMsgFromYaw(yaw);
            obstacle_poses.poses.push_back(p);
            bbs_num++;
        }else{
            std::cout << "rejected" << std::endl;
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

    std::cout << "remove points size: " << remove_indices->indices.size() << std::endl;
    CloudXYZINPtr obstacle_removed_cloud_ptr = CloudXYZINPtr(new CloudXYZIN);
    obstacle_removed_cloud_ptr->header = cloud_ptr->header;
    pcl::ExtractIndices<PointXYZIN> extract;
    extract.setInputCloud(cloud_ptr);
    extract.setIndices(remove_indices);
    extract.setNegative(true);
    extract.filter(*obstacle_removed_cloud_ptr);
    obstacle_removed_cloud_pub.publish(*obstacle_removed_cloud_ptr);

    obstacle_pose_pub.publish(obstacle_poses);

    std::cout << "time: " << ros::Time::now().toSec() - start_time << "[s]" << std::endl;
}

void ObstacleDetectorPCA::get_euclidean_cluster_indices(std::vector<pcl::PointIndices>& cluster_indices)
{
    // set all z of points to 0
    int size = cloud_ptr->points.size();
    std::vector<double> z(size);
    for(int i=0;i<size;i++){
        z[i] = cloud_ptr->points[i].z;
        cloud_ptr->points[i].z = 0.0;
    }

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

    // restore z
    for(int i=0;i<size;i++){
        cloud_ptr->points[i].z = z[i];
    }
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
    double min_z = cluster->points[0].z;
    double max_z = cluster->points[0].z;
    for(auto& pt : cluster->points){
        ave_x += pt.x;
        ave_y += pt.y;
        ave_z += pt.z;
        min_z = std::min(min_z, (double)pt.z);
        max_z = std::max(max_z, (double)pt.z);
    }
    ave_x /= cluster_size;
    ave_y /= cluster_size;
    ave_z /= cluster_size;
    centroid << ave_x, ave_y, ave_z;
    double sigma_xx = 0;
    double sigma_xy = 0;
    double sigma_yy = 0;
    for(auto& pt : cluster->points){
        sigma_xx += (pt.x - ave_x) * (pt.x - ave_x);
        sigma_xy += (pt.x - ave_x) * (pt.y - ave_y);
        sigma_yy += (pt.y - ave_y) * (pt.y - ave_y);
    }
    sigma_xx /= cluster_size;
    sigma_xy /= cluster_size;
    sigma_yy /= cluster_size;
    Eigen::Matrix2d cov_mat;
    cov_mat << sigma_xx, sigma_xy,
               sigma_xy, sigma_yy;
    Eigen::EigenSolver<Eigen::Matrix2d> es(cov_mat);
    Eigen::Vector2d eigen_values = es.eigenvalues().real();
    Eigen::Matrix2d eigen_vectors = es.eigenvectors().real();
    int larger_index = 0;
    if(eigen_values(0) > eigen_values(1)){
        larger_index = 0;
    }else{
        larger_index = 1;
    }
    Eigen::Vector2d first_component_vector = eigen_vectors.col(larger_index);
    double min_inner_product_f = 0;
    double max_inner_product_f = 0;
    Eigen::Vector2d second_component_vector = eigen_vectors.col(1 - larger_index);
    double min_inner_product_s = 0;
    double max_inner_product_s = 0;
    for(const auto& pt : cluster->points){
        Eigen::Vector2d pt_v(pt.x - ave_x, pt.y - ave_y);
        double inner_product_f = pt_v.dot(first_component_vector);
        min_inner_product_f = std::min(min_inner_product_f, inner_product_f);
        max_inner_product_f = std::max(max_inner_product_f, inner_product_f);
        double inner_product_s = pt_v.dot(second_component_vector);
        min_inner_product_s = std::min(min_inner_product_s, inner_product_s);
        max_inner_product_s = std::max(max_inner_product_s, inner_product_s);
    }
    scale(0) = fabs(max_inner_product_f) + fabs(min_inner_product_f);
    scale(1) = fabs(max_inner_product_s) + fabs(min_inner_product_s);
    scale(2) = max_z - min_z;
    yaw = atan2(first_component_vector(1), first_component_vector(0));
    // std::cout << yaw << "[rad]" << std::endl;
}

bool ObstacleDetectorPCA::is_human_cluster(const Eigen::Vector3d& centroid, const Eigen::Vector3d& scale)
{
    // return MIN_HEIGHT < scale(2) && scale(2) < MAX_HEIGHT
    //        && MIN_WIDTH < scale(0) && scale(0) < MAX_WIDTH
    //        && MIN_LENGTH < scale(1) && scale(1) < MAX_LENGTH;
    return (MIN_HEIGHT < scale(2) && scale(2) < MAX_HEIGHT// walking
           && 0.7 < scale(0) && scale(0) < 1.3
           && 0.3 < scale(1) && scale(1) < 0.9)
           || (MIN_HEIGHT < scale(2) && scale(2) < MAX_HEIGHT// stopping
           && 0.4 < scale(0) && scale(0) < 0.7
           && 0.2 < scale(1) && scale(1) < 0.5)
           || (MIN_HEIGHT < scale(2) && scale(2) < MAX_HEIGHT// other
           && 0.5 < scale(0) && scale(0) < 0.9
           && 0.5 < scale(1) && scale(1) < 0.9);
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
