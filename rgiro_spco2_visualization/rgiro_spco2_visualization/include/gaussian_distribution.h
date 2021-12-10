#ifndef RVIZ_MARKER_LIB_GAUSSIAN_H
#define RVIZ_MARKER_LIB_GAUSSIAN_H

#include <ros/ros.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Vector3.h>
#include <std_msgs/ColorRGBA.h>
#include <sensor_msgs/PointCloud2.h>
#include <rgiro_spco2_visualization_msgs/GaussianDistributions.h>
#include <rgiro_spco2_visualization_msgs/GaussianService.h>
#include <visualization_msgs/MarkerArray.h>
#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <nav_msgs/GetMapRequest.h>
#include <nav_msgs/GetMapResponse.h>
#include <modules/animation.h>

namespace gaussian_distribution {

    class GaussianDistribution {

        public:

            GaussianDistribution(const ros::NodeHandle &private_nh = ros::NodeHandle("~"));

        protected:
            ros::NodeHandle m_nh;
            ros::Publisher m_pub;
            ros::Subscriber m_sub;
            ros::ServiceServer m_service;

            std::string m_frame;

            Animation m_animation;

            double m_threshold;
            double m_resolution;
            int m_frame_rate;
            double m_radius;
            double m_cycle_time;

            void call_back(const rgiro_spco2_visualization_msgs::GaussianDistributions::ConstPtr &msg);

            void create_marker(pcl::PointCloud<pcl::PointXYZRGBA> &cloud, double mu_x, double mu_y, double sigma_x, double sigma_y,
                               double sigma_xy, double probability, int r, int g, int b);

            void generate_distribution(const rgiro_spco2_visualization_msgs::GaussianDistributions &msg);

            bool
            service_call_back(rgiro_spco2_visualization_msgs::GaussianService::Request &request,
                              rgiro_spco2_visualization_msgs::GaussianService::Response &response);

    };
}

#endif //RVIZ_MARKER_LIB_GAUSSIAN_H
