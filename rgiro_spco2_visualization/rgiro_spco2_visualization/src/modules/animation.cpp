#include <modules/animation.h>
#include <ros/package.h>

#define PI 3.14159265359

namespace gaussian_distribution {

    using pcl::PointCloud;
    using pcl::PointXYZRGBA;
    using rgiro_spco2_visualization_msgs::GaussianDistribution;
    using rgiro_spco2_visualization_msgs::GaussianDistributions;

    Animation::Animation() : m_threshold(0), m_resolution(0), m_radius(0) {
    }

    Animation::Animation(std::string &frame, double threshold, double resolution, double radius) :
            m_frame(frame), m_threshold(threshold), m_resolution(resolution), m_radius(radius) {
        m_pub = m_nh.advertise<sensor_msgs::PointCloud2>("gaussian_out", 1, true);
    }

    void Animation::animation(const GaussianDistributions &msg, double cycle_time, int frame_rate) {
        double frame_time = cycle_time / (cycle_time * frame_rate);
        if(cycle_time<1){
            frame_time = cycle_time / (frame_rate);
        }
        double average_time = 0;
        std::vector<std::pair<GaussianDistribution, GaussianDistribution>> distributions;
        create_pair(distributions, m_before_msg, msg);
        for (int i = 0; i < frame_rate; ++i) {
            clock_t start = clock();
            generate_distribution(distributions, i, frame_rate);
            clock_t end = clock();
            double time = static_cast<double>(end - start) / (CLOCKS_PER_SEC);
            if (time < frame_time) {
                ros::Duration(frame_time - time).sleep();
            }
            average_time += time;
        }
        average_time /= frame_rate;
        ROS_INFO("[Animation] FrameRate: %.lfFPS(%d)", 1 / average_time, frame_rate);
        m_before_msg = msg;
    }

    void Animation::create_pair(std::vector<std::pair<GaussianDistribution, GaussianDistribution>> &distributions,
                                const GaussianDistributions &before_msg, const GaussianDistributions &msg) {

        int before_size = int(before_msg.distributions.size());
        int next_size = int(msg.distributions.size());
        int max_size = std::max(before_size, next_size);
        std::vector<int> pair_id(max_size, -1);
        std::vector<bool> is_pair(next_size, false);

        for (int i = 0; i < before_size; ++i) {
            std::vector<std::pair<double, int>> pairs(next_size);
            auto &first = before_msg.distributions[i];
            for (int k = 0; k < next_size; ++k) {
                auto &second = msg.distributions[k];
                double dx = first.mean_x - second.mean_x;
                double dy = first.mean_y - second.mean_y;
                double distance = dx * dx + dy * dy;
                pairs[k] = std::make_pair(distance, k);
            }
            std::sort(pairs.begin(), pairs.end());
            for (const auto &pair:pairs) {
                int k = pair.second;
                double d=pair.first;
                if (d<1.0 && !is_pair[k]) {
                    pair_id[i] = k;
                    is_pair[k] = true;
                    break;
                }
            }
        }

        for (int i = 0; i < before_size; ++i) {
            if (pair_id[i] != -1) {
                distributions.emplace_back(std::make_pair(before_msg.distributions[i], msg.distributions[pair_id[i]]));
            }
        }
        for (int i = 0; i < next_size; ++i) {
            if (!is_pair[i]) {
                distributions.emplace_back(std::make_pair(GaussianDistribution(), msg.distributions[i]));
            }
        }
    }

    void Animation::generate_distribution(const std::vector<std::pair<GaussianDistribution, GaussianDistribution>> &distributions,
                                          int i, int loop) {

        PointCloud<PointXYZRGBA> cloud;
        for (const auto &pair : distributions) {
            const auto &first = pair.first;
            const auto &second = pair.second;
            PointCloud<PointXYZRGBA> c;
            if (first.probability == 0) {
                create_marker(c,
                              second.mean_x,
                              second.mean_y,
                              second.variance_x,
                              second.variance_y,
                              second.covariance,
                              (second.probability / loop) * i,
                              second.r,
                              second.g,
                              second.b,
                              (255.0 / loop) * i);
            } else {
                create_marker(c,
                              first.mean_x + ((second.mean_x - first.mean_x) / loop) * i,
                              first.mean_y + ((second.mean_y - first.mean_y) / loop) * i,
                              first.variance_x + ((second.variance_x - first.variance_x) / loop) * i,
                              first.variance_y + ((second.variance_y - first.variance_y) / loop) * i,
                              first.covariance + ((second.covariance - first.covariance) / loop) * i,
                              first.probability + ((second.probability - first.probability) / loop) * i,
                              first.r + (double(second.r - first.r) / loop) * i,
                              first.g + (double(second.g - first.g) / loop) * i,
                              first.b + (double(second.b - first.b) / loop) * i,
                              255.0);
            }
            cloud += c;
        }
        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(cloud, cloud_msg);
        cloud_msg.header.frame_id = m_frame;
        cloud_msg.header.stamp = ros::Time::now();
        m_pub.publish(cloud_msg);
    }

    void Animation::create_marker(PointCloud<PointXYZRGBA> &cloud,
                                  double mu_x, double mu_y,
                                  double sigma_x, double sigma_y, double sigma_xy, double probability,
                                  double r, double g, double b, double a) {
        double sigma_x_y = sigma_x * sigma_y;
        double rho = sigma_xy / (sigma_x_y);
        double sigma_x2 = sigma_x * sigma_x;
        double sigma_y2 = sigma_y * sigma_y;
        double p1 = -2 * (1 - rho * rho);
        double p2 = 2 * PI * sigma_x_y * std::sqrt(1 - rho * rho);


        double max_prob = 0;
        double min_prob = INT_MAX;
        double range_min_x = mu_x - std::min(3 * sigma_x, m_radius);
        double range_max_x = mu_x + std::min(3 * sigma_x, m_radius);
        double range_min_y = mu_y - std::min(3 * sigma_y, m_radius);
        double range_max_y = mu_y + std::min(3 * sigma_y, m_radius);
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
            point.r = std::max(0, int(r - v));
            point.g = std::max(0, int(g - v));
            point.b = std::max(0, int(b - v));
            point.r = std::min(point.r + 20, 255);
            point.g = std::min(point.g + 20, 255);
            point.b = std::min(point.b + 20, 255);
            point.a = std::max(0, int(a * point.z));
        }
        for (auto &point:cloud) {
            point.z = float(point.z * probability);
        }
        cloud.points.shrink_to_fit();
    }

}
