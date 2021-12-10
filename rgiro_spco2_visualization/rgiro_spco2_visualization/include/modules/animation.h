#ifndef MODULES_GAUSSIAN_H
#define MODULES_GAUSSIAN_H

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

namespace gaussian_distribution {

    class Animation {

        public:


            Animation();

            Animation(std::string &frame, double threshold, double resolution, double radius);

            void animation(const rgiro_spco2_visualization_msgs::GaussianDistributions &msg, double cycle_time, int frame_rate);

        protected:
            ros::NodeHandle m_nh;
            ros::Publisher m_pub;
            std::string m_frame;

            double m_threshold;
            double m_resolution;
            double m_radius;

            rgiro_spco2_visualization_msgs::GaussianDistributions m_before_msg;

            void generate_distribution(const std::vector<std::pair<rgiro_spco2_visualization_msgs::GaussianDistribution,
                    rgiro_spco2_visualization_msgs::GaussianDistribution>> &distributions, int i, int loop);

            void create_pair(
                    std::vector<std::pair<rgiro_spco2_visualization_msgs::GaussianDistribution, rgiro_spco2_visualization_msgs::GaussianDistribution>> &distributions,
                    const rgiro_spco2_visualization_msgs::GaussianDistributions &before_msg, const rgiro_spco2_visualization_msgs::GaussianDistributions &msg);

            void
            create_marker(pcl::PointCloud<pcl::PointXYZRGBA> &cloud, double mu_x, double mu_y, double sigma_x, double sigma_y, double sigma_xy,
                          double probability, double r, double g, double b, double a);
    };
}

#endif //RVIZ_MARKER_LIB_GAUSSIAN_H
