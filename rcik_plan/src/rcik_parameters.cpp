/* Author: Mincheul Kang */

#include <rcik_plan/rcik_parameters.h>

RcikParameters::RcikParameters()
{
    smoothness_vel_cost_weight_ = 0.0;
    smoothness_acc_cost_weight_ = 0.0;
    smoothness_jerk_cost_weight_ = 0.0;
    endPose_cost_weight_ = 0.5;
    obstacle_cost_weight_ = 1.0;

    gaussian_weight_ = 0.1;
    min_clearence_ = 0.3;
    min_singularity_value_ = 0.005;

    use_checking_singularity_ = false;
    use_checking_velocity_limit_ = true;
}

RcikParameters::~RcikParameters()
{
}