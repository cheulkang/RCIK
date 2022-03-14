/* Author: Mincheul Kang */

#include <ros/ros.h>
#include <torm/torm_ik_solver.h>
#include <torm/torm_debug.h>
#include <rcik_plan/npy.hpp>
#include <rcik_plan/rcik_planning.h>
#include <rcik_plan/rcik_parameters.h>
#include <rcik_plan/rcik_problem.h>

// MoveIt!
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/planning_interface/planning_interface.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/kinematic_constraints/utils.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <moveit_msgs/PlanningScene.h>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <moveit_msgs/AttachedCollisionObject.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/collision_distance_field/collision_world_hybrid.h>
#include <moveit/collision_distance_field/collision_detector_allocator_hybrid.h>
#include <moveit/robot_state/conversions.h>

#include <tf/tf.h>
#include <visualization_msgs/Marker.h>
#include <eigen_conversions/eigen_kdl.h>

void initParameters(RcikParameters &params_){
    params_.smoothness_vel_cost_weight_ = 0.5;
    params_.smoothness_acc_cost_weight_ = 0.5;
    params_.smoothness_jerk_cost_weight_ = 0.0;
    params_.endPose_cost_weight_ = 300.0;
    params_.obstacle_cost_weight_ = 1.0;
    params_.gaussian_weight_ = 0.07;
    params_.min_clearence_ = 0.3;
    params_.time_generation_ = 0.02;
    params_.time_duration_ = 0.03;
    params_.min_singularity_value_ = 0.005;

    params_.use_checking_singularity_ = false;
    params_.use_checking_velocity_limit_ = true;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "pathwise");
    ros::NodeHandle node_handle("~");

    robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
    robot_model::RobotModelPtr robot_model = robot_model_loader.getModel();
    robot_state::RobotStatePtr robot_state(new robot_state::RobotState(robot_model));
    planning_scene::PlanningScenePtr planning_scene(new planning_scene::PlanningScene(robot_model));
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;

    // initialize planning interface
    std::map<std::string, moveit_msgs::CollisionObject> collision_objects_map = planning_scene_interface_.getObjects();
    std::vector<moveit_msgs::CollisionObject> v_co;
    for(auto& kv : collision_objects_map){
        kv.second.operation = kv.second.REMOVE;
        v_co.push_back(kv.second);
    }
    planning_scene_interface_.applyCollisionObjects(v_co);
    ros::Duration(1.0).sleep();

    RcikProblem prob(planning_scene);

    const std::string PLANNING_GROUP = prob.getPlanningGroup();
    const robot_state::JointModelGroup *joint_model_group = robot_state->getJointModelGroup(PLANNING_GROUP);
    std::vector<std::string> indices = joint_model_group->getActiveJointModelNames();
    robot_model::JointBoundsVector joint_bounds;
    joint_bounds = joint_model_group->getActiveJointModelsBounds();
    unsigned int num_dof = (unsigned int)joint_bounds.size();

    collision_objects_map = planning_scene_interface_.getObjects();
    for(auto& kv : collision_objects_map){
        planning_scene->processCollisionObjectMsg(kv.second);
    }

    robot_state::RobotState state = planning_scene->getCurrentState();
    std::vector<std::string> defaultJoints = prob.getDefaultSettingJoints();
    std::vector<double> defaultValues = prob.getDefaultSettingValues();
    for(uint i = 0; i < defaultJoints.size(); i++)
        state.setJointPositions(defaultJoints[i], &defaultValues[i]);
    planning_scene->setCurrentState(state);
    state.update();

    torm::TormIKSolver iksolver(PLANNING_GROUP, planning_scene, prob.getBaseLink(), prob.getTipLink());
    torm::TormDebug debug(planning_scene, prob.getFixedFrame());

    //// visualize input poses
    const Eigen::Isometry3d& baseT = planning_scene->getFrameTransform(prob.getBaseLink());
    KDL::Frame baseKDL;
    tf::transformEigenToKDL(baseT, baseKDL);

    //// Parameter
    RcikParameters params;
    initParameters(params);

    //// planner
    std::unique_ptr<RcikPlanning> planner;
    planner.reset(new RcikPlanning(planning_scene, PLANNING_GROUP, iksolver, debug, joint_bounds, params, state));

    //// start configuration
    KDL::JntArray q_c(num_dof);
    std::vector<double> s_conf = prob.getStartConfiguration();
    for (uint j = 0; j < num_dof; j++) {
        q_c(j) = s_conf[j];
    }

    Eigen::VectorXd q_cur(num_dof);
    for (int i = 0; i < num_dof; i++){
        q_cur(i) = q_c(i);
    }

    planner->resetTrajectory(q_cur);

    //// Occupancy Grid
    planner->updateEnvironment(prob.getEnvironment());
    std::string model_path = prob.getDeepModel();
    planner->initializeModel(model_path, num_dof);

    //// Planning
    // visualize trajectory
    std::vector<std::string> indices_vis = indices;
    std::vector<double> values_vis;
    for(uint i = 0; i < defaultJoints.size(); i++){
        indices_vis.push_back(defaultJoints[i]);
        values_vis.push_back(defaultValues[i]);
    }

    std::vector<KDL::JntArray> confs;
    std::vector<double> v_qc(num_dof, 0.0);
    for (uint j = 0; j < iksolver.getDoF(); j++) {
        v_qc[j] = q_c(j);
    }
    for(int j = 0; j < defaultJoints.size(); j++){
        v_qc.push_back(values_vis[j]);
    }
    debug.visualizeConfiguration(indices_vis, v_qc);
    confs.push_back(q_c);

    // keyboard
    KDL::Frame currentPose;
    iksolver.fkSolver(q_c, currentPose);
    KDL::Frame goalPose;
    iksolver.fkSolver(q_c, goalPose);

    int mode = 0;
    double t_trans = 0.005;
    double t_rots = 0.03;
    bool m_method = true;

    geometry_msgs::Point p1, p2, p3;
    p1.x = currentPose.p.data[0] + baseKDL.p.data[0];
    p1.y = currentPose.p.data[1] + baseKDL.p.data[1];
    p1.z = currentPose.p.data[2] + baseKDL.p.data[2];

    while (1)
    {
        char c;
        std::cin >> c;
        if (c == 'a'){
            std::cout << "x" << std::endl;
            mode = 0;
        }
        else if (c == 's'){
            std::cout << "y" << std::endl;
            mode = 1;
        }
        else if (c == 'd'){
            std::cout << "z" << std::endl;
            mode = 2;
        }
        else if (c == 'f'){
            std::cout << "roll" << std::endl;
            mode = 3;
        }
        else if (c == 'g'){
            std::cout << "pitch" << std::endl;
            mode = 4;
        }
        else if (c == 'h'){
            std::cout << "yaw" << std::endl;
            mode = 5;
        }
        else if (c == 'k'){
            // calcuate the goal pose
            goalPose = currentPose;
            if(mode < 3){
                goalPose.p.data[mode] -= t_trans;
            }
            else{
                double roll, pitch, yaw;
                goalPose.M.GetRPY(roll, pitch, yaw);

                if(mode == 3){
                    goalPose.M = goalPose.M.RPY(roll - t_rots, pitch, yaw);
                }
                else if(mode == 4){
                    goalPose.M = goalPose.M.RPY(roll, pitch - t_rots, yaw);
                }
                else{
                    goalPose.M = goalPose.M.RPY(roll, pitch, yaw - t_rots);
                }
            }
        }
        else if (c == 'l'){
            // calcuate the goal pose
            goalPose = currentPose;
            if(mode < 3){
                goalPose.p.data[mode] += t_trans;
            }
            else{
                double roll, pitch, yaw;
                goalPose.M.GetRPY(roll, pitch, yaw);

                if(mode == 3){
                    goalPose.M = goalPose.M.RPY(roll + t_rots, pitch, yaw);
                }
                else if(mode == 4){
                    goalPose.M = goalPose.M.RPY(roll, pitch + t_rots, yaw);
                }
                else{
                    goalPose.M = goalPose.M.RPY(roll, pitch, yaw + t_rots);
                }
            }
        }
        else if(c == 'z'){
            break;
        }
        else if(c == 'v'){
            m_method ^= m_method;
        }

        if(c == 'k' || c == 'l'){
            double roll, pitch, yaw;
            goalPose.M.GetRPY(roll, pitch, yaw);

            int res;
            if(m_method){
                std::cout << "SDF: ";
                res = planner->findIKWithSDF(goalPose);
            }
            else{
                std::cout << "CCPN: ";
                res = planner->findIKWithCCPN(goalPose);
            }
            std::cout << goalPose.p.data[0] << " " << goalPose.p.data[1] << " " << goalPose.p.data[2] << " ";
            std::cout << roll << " " << pitch << " " << yaw << std::endl;

            if(res <= 0){
                std::cout << "Fail!!!" << std::endl;
                continue;
            }

            Eigen::VectorXd qq = planner->trajectory_.row(0);
            planner->updateTrajectory(qq);

            std::vector<double> t;
            for(int j = 0; j < num_dof; j++){
                t.push_back(qq(j));
                q_c(j) = qq(j);
            }
            for(int j = 0; j < defaultJoints.size(); j++){
                t.push_back(values_vis[j]);
            }
            confs.push_back(q_c);

            //// line visualization
            p2.x = goalPose.p.data[0] + baseKDL.p.data[0];
            p2.y = goalPose.p.data[1] + baseKDL.p.data[1];
            p2.z = goalPose.p.data[2] + baseKDL.p.data[2];

            debug.visualizeEndEffectorPose_line(p1, p2, 0);

            KDL::Frame end_effector_pose;
            iksolver.fkSolver(q_c, end_effector_pose);
            p3.x = end_effector_pose.p.data[0]+ baseKDL.p.data[0];
            p3.y = end_effector_pose.p.data[1]+ baseKDL.p.data[1];
            p3.z = end_effector_pose.p.data[2]+ baseKDL.p.data[2];

            debug.visualizeEndEffectorPose_line(p1, p3, 2);
            debug.publishEndEffectorPose_line();

            debug.visualizeConfiguration(indices_vis, t);
            iksolver.fkSolver(q_c, currentPose);
            p1 = p3;
        }
    }

    return 0;
}