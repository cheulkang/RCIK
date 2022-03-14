/* Author: Mincheul Kang */

#ifndef RCIK_PARAMETERS_H_
#define RCIK_PARAMETERS_H_

#include <ros/ros.h>

class RcikParameters
{
public:
    RcikParameters();
    virtual ~RcikParameters();

    RcikParameters getNonConstParams(RcikParameters params);

public:
    double smoothness_vel_cost_weight_;
    double smoothness_acc_cost_weight_;
    double smoothness_jerk_cost_weight_;
    double obstacle_cost_weight_;
    double endPose_cost_weight_;
    double gaussian_weight_;
    double min_clearence_;
    double time_duration_;
    double time_generation_;
    double min_singularity_value_;

    bool use_checking_singularity_;
    bool use_checking_velocity_limit_;
};

#endif