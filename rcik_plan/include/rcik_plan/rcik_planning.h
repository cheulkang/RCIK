/* Author: Mincheul Kang */

#ifndef RCIK_PLANNING_H
#define RCIK_PLANNING_H

#include <torm/torm_ik_solver.h>
#include <torm/torm_debug.h>
#include <torm/torm_utils.h>
#include <rcik_plan/rcik_parameters.h>

#include <moveit/robot_model/robot_model.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/collision_distance_field/collision_robot_hybrid.h>
#include <moveit/collision_distance_field/collision_world_hybrid.h>

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <vector>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <trac_ik/trac_ik.hpp>
#include <numeric>
#include <random>

#include <torch/script.h>

class RcikPlanning
{
public:
    RcikPlanning(const planning_scene::PlanningSceneConstPtr& planning_scene,
                 const std::string& planning_group,
                 torm::TormIKSolver& iksolver,
                 torm::TormDebug& debug,
                 robot_model::JointBoundsVector& bounds,
                 const RcikParameters& parameters,
                 const moveit::core::RobotState& state);
    virtual ~RcikPlanning();

    void initializeSDF();
    void initializeModel(std::string &model_path, int num);
    KDL::JntArray fGaussian(KDL::JntArray q);
    int findIKWithSDF(const KDL::Frame &target_pose);
    int findIKWithCCPN(const KDL::Frame &target_pose);

    bool findSample(const KDL::JntArray& q_init, const KDL::Frame& p_in, KDL::JntArray& q_out, bool random=false);
    double getSmoothnessCost();
    double getEndPoseCost(const KDL::Frame &target_pose, bool isGrad);
    double getCollisionCost();

    void setRobotStateFromPoint();
    void computeJointProperties();
    void performForwardKinematics();

    void updateEnvironment(const std::vector<float> &env);
    void resetTrajectory(Eigen::VectorXd &q);
    void updateTrajectory(Eigen::VectorXd &q);

    bool isCurrentTrajectoryCollisionFree() const;
    bool jointVelocityLimitCheck(const KDL::JntArray &q);
    bool kinematicSingularityCheck(const KDL::JntArray &q);

    int num_trajectory_;
    Eigen::MatrixXd trajectory_;
private:
    std::string planning_group_;
    planning_scene::PlanningSceneConstPtr planning_scene_;
    torm::TormIKSolver &iksolver_;
    torm::TormDebug &debug_;
    const RcikParameters& parameters_;

    int num_joints_;
    int start_collision_;
    int end_collision_;

    const moveit::core::JointModelGroup* joint_model_group_;
    moveit::core::RobotState state_;
    const moveit::core::RobotModelConstPtr& kmodel_;
    collision_detection::CollisionWorldHybrid* hy_world_;
    collision_detection::CollisionRobotHybrid* hy_robot_;
    collision_detection::GroupStateRepresentationPtr gsr_;

    std::vector<double> vel_limit_;
    std::vector<std::string> joint_names_;
    std::map<std::string, std::map<std::string, bool> > joint_parent_map_;

    std::vector<bool> joint_kinds_;

    int num_collision_points_;
    std::vector<int> point_is_in_collision_;
    int max_collision_point_;

    std::vector<std::string>  collision_point_joint_names_;
    std::vector<EigenSTL::vector_Vector3d> collision_point_pos_eigen_;
    std::vector<std::vector<double> > collision_point_potential_;
    EigenSTL::vector_Vector3d joint_axes_;
    EigenSTL::vector_Vector3d joint_positions_;

    std::vector<double> variance_;

    std::default_random_engine generator_;
    collision_detection::AllowedCollisionMatrix acm_;

    void registerParents(const moveit::core::JointModel* model);

    inline double getPotential(double field_distance, double radius, double clearence) {
        double d = field_distance - radius;
        double potential = clearence - d;

        if(potential < 0.0){
            return 0.0;
        }
        return potential;
    }

    inline bool isParent(const std::string& childLink, const std::string& parentLink) const {
        if (childLink == parentLink)
        {
            return true;
        }

        if (joint_parent_map_.find(childLink) == joint_parent_map_.end())
        {
            // ROS_ERROR("%s was not in joint parent map! for lookup of %s", childLink.c_str(), parentLink.c_str());
            return false;
        }
        const std::map<std::string, bool>& parents = joint_parent_map_.at(childLink);
        return (parents.find(parentLink) != parents.end() && parents.at(parentLink));
    }

    torch::jit::script::Module module_;
    float t_env_[40*40*40];
};

#endif 