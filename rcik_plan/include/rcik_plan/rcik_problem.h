/* Author: Mincheul Kang */

#ifndef RCIK_PROBLEM_H
#define RCIK_PROBLEM_H

#include <ros/ros.h>
#include <ros/package.h>

#include <fstream>
#include <trac_ik/trac_ik.hpp>
#include <moveit_msgs/PlanningScene.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

#include <torm/torm_utils.h>
#include <rcik_plan/npy.hpp>

class RcikProblem
{
public:
    RcikProblem(const planning_scene::PlanningSceneConstPtr& planning_scene);
    ~RcikProblem();

    void setCollisionObjects();
    void removeCollisionObjects(std::map<std::string, moveit_msgs::CollisionObject> &collision_objects_map);
    void setOccupancyGrid();
    moveit_msgs::CollisionObject makeCollisionObject(std::string name, double x, double y, double z,
                                                     double roll, double pitch, double yaw,
                                                     double size_x, double size_y, double size_z);
    std::string getPlanningGroup();
    std::string getFixedFrame();
    std::string getBaseLink();
    std::string getTipLink();
    std::vector<std::string> getDefaultSettingJoints();
    std::vector<double> getDefaultSettingValues();
    std::vector<double> getStartConfiguration();
    std::vector<float> getEnvironment();
    std::string getDeepModel();
private:
    ros::NodeHandle                                             nh_;
    std::string                                                 fixed_frame_;
    std::string                                                 planning_group_;
    std::string                                                 planning_base_link_;
    std::string                                                 planning_tip_link_;
    std::vector<double>                                         start_config_;
    std::vector<double>                                         start_pose_;
    std::string                                                 occupancy_grid_;
    std::string                                                 self_occupancy_grid_;
    std::vector<std::string>                                    default_setting_joints_;
    std::vector<double>                                         default_setting_values_;
    std::vector<moveit_msgs::CollisionObject>                   collision_objects_;
    std::vector<KDL::Frame>                                     target_poses_;
    std::string                                                 deep_model_;
    std::string                                                 package_path_;

    planning_scene::PlanningSceneConstPtr                       planning_scene_;
    moveit::planning_interface::PlanningSceneInterface          planning_scene_interface_;

    std::vector<float>                                         env_;
};

#endif