#include <gaussian_distribution.h>
#include <ros/package.h>
#include <nav_msgs/GetMap.h>

#define PI 3.14159265359

namespace gaussian_distribution {

    using pcl::PointCloud;
    using pcl::PointXYZRGBA;

    GaussianDistribution::GaussianDistribution(const ros::NodeHandle &private_nh) :
            m_frame("map") {

        private_nh.param("frame", m_frame, m_frame);
        private_nh.param("threshold", m_threshold, 0.00001);
        private_nh.param("resolution", m_resolution, 0.05);
        private_nh.param("frame_rate", m_frame_rate, 30);
        private_nh.param("radius", m_radius, 2.0);
        private_nh.param("cycle_time", m_cycle_time, 1.0);

        m_animation = Animation(m_frame, m_threshold, m_resolution, m_radius);

        ros::NodeHandle nh = ros::NodeHandle("~");
        m_sub = m_nh.subscribe<rgiro_spco2_visualization_msgs::GaussianDistributions>("gaussian_in", 10, &GaussianDistribution::call_back, this);

        m_service = m_nh.advertiseService("gaussian_srv", &GaussianDistribution::service_call_back, this);
        m_pub = m_nh.advertise<sensor_msgs::PointCloud2>("gaussian_out", 1, true);
        ros::Duration(0.1).sleep();
    }

    bool GaussianDistribution::service_call_back(rgiro_spco2_visualization_msgs::GaussianService::Request &request,
                                                 rgiro_spco2_visualization_msgs::GaussianService::Response &response) {
        m_animation.animation(request.distributions, m_cycle_time, m_frame_rate);
        return true;
    }

    void GaussianDistribution::call_back(const rgiro_spco2_visualization_msgs::GaussianDistributions::ConstPtr &msg) {
        generate_distribution(*msg);
    }

    void GaussianDistribution::generate_distribution(const rgiro_spco2_visualization_msgs::GaussianDistributions &msg) {
        clock_t start = clock();
        auto distributions = msg.distributions;
        PointCloud<PointXYZRGBA> cloud;
        for (const auto &distribution : distributions) {
            PointCloud<PointXYZRGBA> c;
            create_marker(c, distribution.mean_x, distribution.mean_y,
                          distribution.variance_x, distribution.variance_y, distribution.covariance, distribution.probability,
                          distribution.r, distribution.g, distribution.b);
            cloud += c;
        }
        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(cloud, cloud_msg);
        cloud_msg.header.frame_id = m_frame;
        cloud_msg.header.stamp = ros::Time::now();
        m_pub.publish(cloud_msg);

        clock_t end = clock();
        const double time = static_cast<double>(end - start) / (CLOCKS_PER_SEC);
        printf("elapsed time %lf[ms]\n", time);
    }

    void GaussianDistribution::create_marker(PointCloud<PointXYZRGBA> &cloud,
                                             const double mu_x, const double mu_y,
                                             const double sigma_x, const double sigma_y, const double sigma_xy, const double probability,
                                             const int r, const int g, const int b) {
        double sigma_x_y = sigma_x * sigma_y;
        double rho = sigma_xy / (sigma_x_y);
        double sigma_x2 = sigma_x * sigma_x;
        double sigma_y2 = sigma_y * sigma_y;
        double p1 = -2 * (1 - rho * rho);
        double p2 = 2 * PI * sigma_x_y * std::sqrt(1 - rho * rho);

        double max_prob = 0;
        double min_prob = INT_MAX;
        double range_min_x = mu_x - 3 * sigma_x;
        double range_max_x = mu_x + 3 * sigma_x;
        double range_min_y = mu_y - 3 * sigma_y;
        double range_max_y = mu_y + 3 * sigma_y;
        cloud.points.reserve(int((((range_max_x - mu_x) / m_resolution) * 2) * (((range_max_x - mu_x) / m_resolution) * 2)));

        for (double x = range_min_x; x < range_max_x; x += m_resolution) {
            for (double y = range_min_y; y < range_max_y; y += m_resolution) {
                double f = std::exp(((((x - mu_x) * (x - mu_x)) / sigma_x2) + (((y - mu_y) * (y - mu_y)) / sigma_y2)
                                     - (2 * rho * (((x - mu_x) * (y - mu_y)) / sigma_x_y))) / p1) / p2;
                if (f > m_threshold) {
                    PointXYZRGBA point;
                    point.x = float(x);
                    point.y = float(y);
                    point.z = float(f);
                    point.r = r;
                    point.g = g;
                    point.b = b;
                    cloud.points.emplace_back(point);
                    if (f > max_prob) {
                        max_prob = f;
                    }
                    if (f < min_prob) {
                        min_prob = f;
                    }
                }
            }
        }
        for (auto &point:cloud) {
            point.z = float((point.z - min_prob) / max_prob);
        }
        for (auto &point:cloud) {
            int v = int(255 * (1 - point.z));
            point.r = std::max(0, point.r - v);
            point.g = std::max(0, point.g - v);
            point.b = std::max(0, point.b - v);
            point.r = std::min(point.r + 5, 255);
            point.g = std::min(point.g + 5, 255);
            point.b = std::min(point.b + 5, 255);
            point.a = std::max(0, int(255 * point.z));
        }
        for (auto &point:cloud) {
            point.z = float(point.z * probability);
        }
        cloud.points.shrink_to_fit();
    }

}
