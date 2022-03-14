/* Author: Mincheul Kang */

#include <rcik_plan/rcik_problem.h>
#include <tf/LinearMath/Quaternion.h>

RcikProblem::RcikProblem(const planning_scene::PlanningSceneConstPtr& planning_scene){
    planning_scene_ = planning_scene;

    nh_.getParam("/rcik/fixed_frame", fixed_frame_);
    nh_.getParam("/rcik/planning_group", planning_group_);
    nh_.getParam("/rcik/planning_base_link", planning_base_link_);
    nh_.getParam("/rcik/planning_tip_link", planning_tip_link_);
    nh_.getParam("/rcik/start_config", start_config_);
    nh_.getParam("/rcik/start_pose", start_pose_);
    nh_.getParam("/rcik/occupancy_grid", occupancy_grid_);
    nh_.getParam("/rcik/self_occupancy_grid", self_occupancy_grid_);
    nh_.getParam("/rcik/default_setting_joints", default_setting_joints_);
    nh_.getParam("/rcik/default_setting_values", default_setting_values_);
    nh_.getParam("/rcik/deep_model", deep_model_);

    package_path_ = ros::package::getPath("rcik_plan");

    setCollisionObjects();
    planning_scene_interface_.applyCollisionObjects(collision_objects_);
    setOccupancyGrid();
}

RcikProblem::~RcikProblem() {

}

void RcikProblem::setOccupancyGrid(){
    if(occupancy_grid_ == "" && self_occupancy_grid_ != ""){
        std::vector<unsigned long> shape, self_shape;
        bool fortran_order;

        shape.clear();

        std::vector<double> self_data;
        npy::LoadArrayFromNumpy(package_path_ + "/src/data/occ/" + self_occupancy_grid_, self_shape, fortran_order, self_data);

        std::vector<float> env(self_data.begin(), self_data.end());
        env_ = env;
    }
    else if(occupancy_grid_ == ""){
        std::vector<float> empty(40*40*40, 0.0);
        env_ = empty;
    }
    else{
        std::vector<unsigned long> shape, self_shape;
        bool fortran_order, self_fortan_order;
        std::vector<double> data;

        shape.clear();
        data.clear();

        if(self_occupancy_grid_ == ""){
            npy::LoadArrayFromNumpy(package_path_ + "/src/data/occ/" + occupancy_grid_, shape, fortran_order, data);
        }
        else{
            std::vector<double> self_data;
            npy::LoadArrayFromNumpy(package_path_ + "/src/data/occ/" + self_occupancy_grid_, self_shape, self_fortan_order, self_data);
            npy::LoadArrayFromNumpy(package_path_ + "/src/data/occ/" + occupancy_grid_, shape, fortran_order, data);

            for (int i = 0; i < shape[0]*shape[1]*shape[2]; i++){
                if(self_data[i] == 1.0){
                    data[i] = 1.0;
                }
            }

        }

        std::vector<float> env(data.begin(), data.end());
        env_ = env;
    }
}

void RcikProblem::setCollisionObjects(){
    XmlRpc::XmlRpcValue obs;

    nh_.getParam("/rcik/obstacles", obs);
    for(uint i = 0; i < obs.size(); i++){
        std::string obs_name = "obs" + std::to_string(i);

        collision_objects_.push_back(makeCollisionObject(obs_name, double(obs[i][0]["x"]), double(obs[i][1]["y"]), double(obs[i][2]["z"]),
                                                         double(obs[i][3]["roll"]), double(obs[i][4]["pitch"]), double(obs[i][5]["yaw"]),
                                                         double(obs[i][6]["size_x"]), double(obs[i][7]["size_y"]), double(obs[i][8]["size_z"])));
    }
}

void RcikProblem::removeCollisionObjects(std::map<std::string, moveit_msgs::CollisionObject> &collision_objects_map){
    for(auto& kv : collision_objects_){
        kv.operation = kv.REMOVE;
    }
    planning_scene_interface_.applyCollisionObjects(collision_objects_);
    ros::Duration(1.0).sleep();
}

moveit_msgs::CollisionObject RcikProblem::makeCollisionObject(std::string name, double x, double y, double z,
                                                              double roll, double pitch, double yaw,
                                                              double size_x, double size_y, double size_z){
    moveit_msgs::CollisionObject co;

    co.id = name;
    co.header.frame_id = fixed_frame_;

    co.primitives.resize(1);
    co.primitives[0].type = co.primitives[0].BOX;
    co.primitives[0].dimensions.resize(3);
    co.primitives[0].dimensions[0] = size_x;
    co.primitives[0].dimensions[1] = size_y;
    co.primitives[0].dimensions[2] = size_z;

    co.primitive_poses.resize(1);
    co.primitive_poses[0].position.x = x;
    co.primitive_poses[0].position.y = y;
    co.primitive_poses[0].position.z = z;

    tf::Quaternion q;
    q.setRPY(roll, pitch, yaw);

    co.primitive_poses[0].orientation.w = q.w();
    co.primitive_poses[0].orientation.x = q.x();
    co.primitive_poses[0].orientation.y = q.y();
    co.primitive_poses[0].orientation.z = q.z();

    co.operation = co.ADD;

    return co;
}

std::string RcikProblem::getPlanningGroup(){
    return planning_group_;
}

std::string RcikProblem::getFixedFrame(){
    return fixed_frame_;
}

std::string RcikProblem::getBaseLink(){
    return planning_base_link_;
}

std::string RcikProblem::getTipLink(){
    return planning_tip_link_;
}

std::vector<std::string> RcikProblem::getDefaultSettingJoints(){
    return default_setting_joints_;
}

std::vector<double> RcikProblem::getDefaultSettingValues(){
    return default_setting_values_;
}

std::vector<double> RcikProblem::getStartConfiguration(){
    return start_config_;
}

std::vector<float> RcikProblem::getEnvironment(){
    return env_;
}

std::string RcikProblem::getDeepModel(){
    return package_path_ + "/src/data/model/" + deep_model_;
}
