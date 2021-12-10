#include <gaussian_distribution.h>

int main(int argc, char **argv) {
    ros::init(argc, argv, "gaussian_distribution");
    const ros::NodeHandle &private_nh = ros::NodeHandle("~");

    gaussian_distribution::GaussianDistribution gaussian_distribution;

    ros::spin();

    return 0;
}

