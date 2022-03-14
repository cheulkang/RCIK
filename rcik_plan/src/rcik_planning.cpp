/* Author: Mincheul Kang */

#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <rcik_plan/rcik_planning.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/planning_scene/planning_scene.h>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/Core>
#include <rcik_plan/npy.hpp>
#include "/usr/local/cuda-10.1/include/cuda_runtime_api.h"

RcikPlanning::RcikPlanning(const planning_scene::PlanningSceneConstPtr& planning_scene,
                           const std::string& planning_group,
                           torm::TormIKSolver& iksolver,
                           torm::TormDebug& debug,
                           robot_model::JointBoundsVector& bounds,
                           const RcikParameters& parameters,
                           const moveit::core::RobotState& state):
        planning_scene_(planning_scene),
        planning_group_(planning_group),
        kmodel_(planning_scene->getRobotModel()),
        iksolver_(iksolver),
        debug_(debug),
        parameters_(parameters),
        state_(state)
{
    for(uint i = 0; i < bounds.size(); i++){
        vel_limit_.push_back(bounds[i][0][0].max_velocity_);
    }

    num_joints_ = iksolver_.getDoF();
    joint_model_group_ = planning_scene_->getRobotModel()->getJointModelGroup(planning_group_);

    num_trajectory_ = 3;
    trajectory_ = Eigen::MatrixXd::Zero(num_trajectory_, num_joints_);

    const std::vector<const moveit::core::JointModel*> joint_models = joint_model_group_->getActiveJointModels();
    for (size_t joint_i = 0; joint_i < joint_models.size(); joint_i++) {
        const moveit::core::JointModel *joint_model = joint_models[joint_i];

        if (joint_model->getType() == moveit::core::JointModel::REVOLUTE) {
            const moveit::core::RevoluteJointModel *revolute_joint =
                    dynamic_cast<const moveit::core::RevoluteJointModel *>(joint_model);
            if (revolute_joint->isContinuous()) {
                joint_kinds_.push_back(true);
            }
            else{
                joint_kinds_.push_back(false);
            }
        }
    }

    initializeSDF();
}

RcikPlanning::~RcikPlanning()
{
    delete hy_robot_;
    delete hy_world_;
}

void RcikPlanning::initializeSDF(){
    double size_x = 2.5, size_y = 2.5, size_z = 5.0;
    double resolution = 0.01;
    const collision_detection::WorldPtr& w = (const collision_detection::WorldPtr &) planning_scene_->getWorld();
    hy_world_ = new collision_detection::CollisionWorldHybrid(w, Eigen::Vector3d(size_x, size_y, size_z),
                                                              Eigen::Vector3d(0.0, 0.0, 0.0),
                                                              false,
                                                              resolution, 0.0, 0.3);
    if (!hy_world_)
    {
        ROS_WARN_STREAM("Could not initialize hybrid collision world from planning scene");
        return;
    }

    std::map<std::string, std::vector<collision_detection::CollisionSphere>> link_body_decompositions;
    hy_robot_ = new collision_detection::CollisionRobotHybrid(kmodel_, link_body_decompositions,
                                                              size_x, size_y, size_z,
                                                              false,
                                                              resolution, 0.0, 0.3);
    if (!hy_robot_)
    {
        ROS_WARN_STREAM("Could not initialize hybrid collision robot from planning scene");
        return;
    }

    collision_detection::CollisionRequest req;
    collision_detection::CollisionResult res;
    req.group_name = planning_group_;
    ros::WallTime wt = ros::WallTime::now();
    collision_detection::AllowedCollisionMatrix acm = planning_scene_->getAllowedCollisionMatrix();

    hy_world_->getCollisionGradients(req, res, *hy_robot_->getCollisionRobotDistanceField().get(), state_,
                                     &acm, gsr_);

    ROS_INFO_STREAM("First coll check took " << (ros::WallTime::now() - wt));
    num_collision_points_ = 0;
    start_collision_ = 0;
    end_collision_ = gsr_->gradients_.size();
    for (size_t i = start_collision_; i < end_collision_; i++) {
        num_collision_points_ += gsr_->gradients_[i].gradients.size();
    }

    collision_point_joint_names_.resize(num_collision_points_);
    collision_point_pos_eigen_.resize(1, EigenSTL::vector_Vector3d(num_collision_points_));
    collision_point_potential_.resize(1, std::vector<double>(num_collision_points_));

    point_is_in_collision_.resize(num_collision_points_);

    joint_axes_.resize(num_joints_);
    joint_positions_.resize(num_joints_);

    std::map<std::string, std::string> fixed_link_resolution_map;
    for (int i = 0; i < num_joints_; i++) {
        joint_names_.push_back(joint_model_group_->getActiveJointModels()[i]->getName());
        registerParents(joint_model_group_->getActiveJointModels()[i]);
        fixed_link_resolution_map[joint_names_[i]] = joint_names_[i];
    }

    for (const moveit::core::JointModel* jm : joint_model_group_->getFixedJointModels())
    {
        if (!jm->getParentLinkModel())  // root joint doesn't have a parent
            continue;

        fixed_link_resolution_map[jm->getName()] = jm->getParentLinkModel()->getParentJointModel()->getName();
    }

    // TODO - is this just the joint_roots_?
    for (size_t i = 0; i < joint_model_group_->getUpdatedLinkModels().size(); i++)
    {
        if (fixed_link_resolution_map.find(
                joint_model_group_->getUpdatedLinkModels()[i]->getParentJointModel()->getName()) ==
            fixed_link_resolution_map.end())
        {
            const moveit::core::JointModel* parent_model = NULL;
            bool found_root = false;

            while (!found_root)
            {
                if (parent_model == NULL)
                {
                    parent_model = joint_model_group_->getUpdatedLinkModels()[i]->getParentJointModel();
                }
                else
                {
                    parent_model = parent_model->getParentLinkModel()->getParentJointModel();
                    for (size_t j = 0; j < joint_names_.size(); j++)
                    {
                        if (parent_model->getName() == joint_names_[j])
                        {
                            found_root = true;
                        }
                    }
                }
            }
            fixed_link_resolution_map[joint_model_group_->getUpdatedLinkModels()[i]->getParentJointModel()->getName()] =
                    parent_model->getName();
        }
    }

    size_t j = 0;
    for (size_t g = start_collision_; g < end_collision_; g++)
    {
        collision_detection::GradientInfo& info = gsr_->gradients_[g];

        for (size_t k = 0; k < info.sphere_locations.size(); k++)
        {
            if (fixed_link_resolution_map.find(info.joint_name) != fixed_link_resolution_map.end())
            {
                collision_point_joint_names_[j] = fixed_link_resolution_map[info.joint_name];
            }
            else
            {
                ROS_ERROR("Couldn't find joint %s!", info.joint_name.c_str());
            }
            j++;
        }
    }
}

void RcikPlanning::initializeModel(std::string &model_path, int num){
    try {
        module_ = torch::jit::load(model_path, at::kCUDA);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
    }

    module_.eval();

    // do run
    at::Tensor t_ddd = torch::from_blob(t_env_, {1, 1, 40, 40, 40}).to(at::kCUDA);

    float dd[num];

    for(int i = 0; i < num; i++)
        dd[i] = 0.0;

    std::vector<torch::jit::IValue> inputs;
    at::Tensor t_dd = torch::from_blob(dd, {1, num});

    inputs.push_back(t_dd.to(at::kCUDA));
    inputs.push_back(t_ddd.clone());

    at::Tensor output = module_.forward(inputs).toTensor();

    cudaDeviceSynchronize();

    for(int i = 0; i < num_joints_; i++){
        variance_.push_back(parameters_.gaussian_weight_ * vel_limit_[i]);
    }

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator_.seed(seed);
}

void RcikPlanning::updateEnvironment(const std::vector<float> &env){
    std::copy(env.begin(), env.end(), t_env_);
}

void RcikPlanning::resetTrajectory(Eigen::VectorXd &q) {
    for (int i = 0; i < num_trajectory_; i++){
        trajectory_.row(i) = q;
    }
}

void RcikPlanning::updateTrajectory(Eigen::VectorXd &q){
    for (int i = num_trajectory_-1; i > 0; i--){
        trajectory_.row(i) = trajectory_.row(i-1);
    }
}

KDL::JntArray RcikPlanning::fGaussian(KDL::JntArray q)
{
    KDL::JntArray q_t(num_joints_);

    for(uint j = 0; j < num_joints_; j++){
        std::normal_distribution<double> distribution(q(j), variance_[j]);
        q_t(j) = distribution(generator_);
    }
    return q_t;
}

bool RcikPlanning::findSample(const KDL::JntArray& q_init, const KDL::Frame& p_in, KDL::JntArray& q_out, bool random) {
    if(random){
        iksolver_.ikSolverCollFree(p_in, q_out);
    }
    else{
        if (!iksolver_.ikSolver(q_init, p_in, q_out)) {
            return false;
        }
    }

    for (int j = 0; j < num_joints_; j++) {
        if(!joint_kinds_[j])
            continue;
        while (1) {
            if (q_out(j) > M_PI)
                q_out(j) -= 2 * M_PI;
            else if (q_out(j) < -M_PI)
                q_out(j) += 2 * M_PI;
            else
                break;
        }
    }

    return jointVelocityLimitCheck(q_out) && kinematicSingularityCheck(q_out);
}

int RcikPlanning::findIKWithSDF(const KDL::Frame &target_pose) {
    double bestCost = 1000000.0;
    Eigen::VectorXd bestQ;

    Eigen::VectorXd pre_q = trajectory_.row(0);

    KDL::JntArray q_c(num_joints_);
    KDL::JntArray q_t(num_joints_);
    KDL::JntArray q_cc(num_joints_);

    for(int j = 0; j < num_joints_; j++){
        q_cc(j) = trajectory_(0, j);
    }

    double sCost, cCost, cost;
    int tries = 0, cands = 0;
    ros::WallTime start_time = ros::WallTime::now();
    std::vector<KDL::JntArray> v_cands;

    while((ros::WallTime::now() - start_time).toSec() < parameters_.time_duration_){
        tries++;
        if(tries == 1)
            q_c = q_cc;
        else
            q_c = fGaussian(q_cc);

        if(!findSample(q_c, target_pose, q_t))
            continue;
        cands++;
        v_cands.push_back(q_t);
        for(int j = 0; j < num_joints_; j++){
            trajectory_(0, j) = q_t(j);
        }

        // compute the cost of smoothness and collision
        sCost = getSmoothnessCost();

        performForwardKinematics();
        cCost = getCollisionCost();

        cost = sCost + cCost;

        // compare the cost with collision checking and then update the best solution
        if(bestCost > cost) {
            if (!isCurrentTrajectoryCollisionFree()) {
                bestCost = cost;
                bestQ = trajectory_.row(0);
            }
        }
    }

    if(cands == 0) {
        trajectory_.row(0) = pre_q;
        return -1;
    }

    if(bestCost != 1000000.0) {
        trajectory_.row(0) = bestQ;
        return 1;
    }
    else{
        trajectory_.row(0) = pre_q;
        return 0;
    }
}

int RcikPlanning::findIKWithCCPN(const KDL::Frame &target_pose) {
    double bestCost = 1000000.0;
    Eigen::VectorXd bestQ;

    Eigen::VectorXd pre_q = trajectory_.row(0);

    KDL::JntArray q_c(num_joints_);
    KDL::JntArray q_t(num_joints_);
    KDL::JntArray q_cc(num_joints_);

    for(int j = 0; j < num_joints_; j++){
        q_cc(j) = trajectory_(0, j);
    }

    std::vector<KDL::JntArray> cands;
    double sCost, cCost, cost;

    ros::WallTime start_time = ros::WallTime::now();
    at::Tensor t_ddd = torch::from_blob(t_env_, {1, 1, 40, 40, 40}).to(at::kCUDA);

    q_c = q_cc;
    if(findSample(q_c, target_pose, q_t)){
        cands.push_back(q_t);
    }

    int tries = 1;
    while((ros::WallTime::now() - start_time).toSec() < parameters_.time_generation_) {
        tries++;
        q_c = fGaussian(q_cc);
        if(!findSample(q_c, target_pose, q_t)){
            continue;
        }
        cands.push_back(q_t);
    }
    uint s_cands = cands.size();
    if(s_cands == 0) {
        trajectory_.row(0) = pre_q;
        return -1;
    }

    float dd[s_cands][num_joints_];
    for (uint k = 0; k < s_cands; k++) {
        // get IK sample using gaussian distribution
        for (int j = 0; j < num_joints_; j++) {
            dd[k][j] = cands[k](j);
        }
    }

    std::vector<torch::jit::IValue> inputs;
    at::Tensor t_dd = torch::from_blob(dd, {s_cands, num_joints_});

    inputs.push_back(t_dd.to(at::kCUDA));
    inputs.push_back(t_ddd.clone());

    at::Tensor output = module_.forward(inputs).toTensor();

    for (uint k = 0; k < s_cands; k++) {
        for (int j = 0; j < num_joints_; j++) {
            trajectory_(0, j) = cands[k](j);
        }
        sCost = getSmoothnessCost();
        cCost = (double)output[k].item<float>();
        cost = sCost + cCost;

        // compare the cost with collision checking and then update the best solution
        if(bestCost > cost) {
            if (!isCurrentTrajectoryCollisionFree()) {
                bestCost = cost;
                bestQ = trajectory_.row(0);
            }
        }

        if((ros::WallTime::now() - start_time).toSec() > parameters_.time_duration_){
            break;
        }
    }

    cudaDeviceSynchronize();

    if(bestCost != 1000000.0) {
        trajectory_.row(0) = bestQ;
        return 1;
    }
    else{
        trajectory_.row(0) = pre_q;
        return 0;
    }
}

double RcikPlanning::getSmoothnessCost() {
    double smoothness_cost = 0.0;

    // joint costs:
    for (int i = 0; i < num_joints_; i++) {
        double v = std::abs(trajectory_(0, i) - trajectory_(1, i));
        if(joint_kinds_[i]){
            if(v > M_PI){
                v = 2*M_PI - v;
            }
        }
        smoothness_cost += parameters_.smoothness_vel_cost_weight_ * v*v;

        double v2 = std::abs(trajectory_(1, i) - trajectory_(2, i));
        if(joint_kinds_[i]){
            if(v2 > M_PI){
                v2 = 2*M_PI - v2;
            }
        }
        smoothness_cost += parameters_.smoothness_acc_cost_weight_ * (v2-v)*(v2-v);
    }

    return smoothness_cost;
}

double RcikPlanning::getEndPoseCost(const KDL::Frame &target_pose, bool isGrad) {
    // forward kinematics
    KDL::JntArray q(num_joints_);
    KDL::JntArray delta_q(num_joints_);
    KDL::Frame endPoses_c;
    KDL::Twist delta_twist;

    q.data = trajectory_.row(0);
    iksolver_.fkSolver(q, endPoses_c);
    delta_twist = diff(endPoses_c, target_pose);
    if(isGrad){
        iksolver_.vikSolver(q, delta_twist, delta_q);
    }

    // L2 distance for delta_twist
    double dd = 0.0;
    for(int j = 0; j < 6; j++){
        dd += delta_twist(j)*delta_twist(j);
    }

    return parameters_.endPose_cost_weight_ * dd;
}

double RcikPlanning::getCollisionCost() {
    double collision_cost = 0.0;

    // collision costs:
    for (int j = 0; j < num_collision_points_; j++) {
        collision_cost += collision_point_potential_[0][j];
    }
    return parameters_.obstacle_cost_weight_ * collision_cost;
}

void RcikPlanning::performForwardKinematics() {
    collision_detection::CollisionRequest req;
    collision_detection::CollisionResult res;
    req.group_name = planning_group_;

    setRobotStateFromPoint();
    hy_world_->getCollisionGradients(req, res, *hy_robot_->getCollisionRobotDistanceField().get(), state_, &acm_, gsr_);
    computeJointProperties();

    // Keep vars in scope
    size_t j = 0;
    for (size_t g = start_collision_; g < end_collision_; g++){
        collision_detection::GradientInfo& info = gsr_->gradients_[g];

        for (size_t k = 0; k < info.sphere_locations.size(); k++) {
            collision_point_pos_eigen_[0][j][0] = info.sphere_locations[k].x();
            collision_point_pos_eigen_[0][j][1] = info.sphere_locations[k].y();
            collision_point_pos_eigen_[0][j][2] = info.sphere_locations[k].z();

            collision_point_potential_[0][j] =
                    getPotential(info.distances[k], info.sphere_radii[k], parameters_.min_clearence_);
            j++;
        }
    }
}

void RcikPlanning::setRobotStateFromPoint() {
    std::vector<double> joint_states;
    for (int j = 0; j < num_joints_; j++) {
        joint_states.push_back(trajectory_(0, j));
    }

    state_.setJointGroupPositions(planning_group_, joint_states);
    state_.update();
}

void RcikPlanning::computeJointProperties() {
    for (int j = 0; j < num_joints_; j++) {
        const moveit::core::JointModel* joint_model = state_.getJointModel(joint_names_[j]);
        const moveit::core::RevoluteJointModel* revolute_joint =
                dynamic_cast<const moveit::core::RevoluteJointModel*>(joint_model);
        const moveit::core::PrismaticJointModel* prismatic_joint =
                dynamic_cast<const moveit::core::PrismaticJointModel*>(joint_model);

        std::string parent_link_name = joint_model->getParentLinkModel()->getName();
        std::string child_link_name = joint_model->getChildLinkModel()->getName();
        Eigen::Affine3d joint_transform =
                state_.getGlobalLinkTransform(parent_link_name) *
                (kmodel_->getLinkModel(child_link_name)->getJointOriginTransform() * (state_.getJointTransform(joint_model)));

        // joint_transform = inverseWorldTransform * jointTransform;
        Eigen::Vector3d axis;

        if (revolute_joint != NULL) {
            axis = revolute_joint->getAxis();
        }
        else if (prismatic_joint != NULL){
            axis = prismatic_joint->getAxis();
        }
        else {
            axis = Eigen::Vector3d::Identity();
        }

        axis = joint_transform * axis;

        joint_axes_[j] = axis;
        joint_positions_[j] = joint_transform.translation();
    }
}

bool RcikPlanning::isCurrentTrajectoryCollisionFree() const {
    bool collision = false;

    std::vector<double> conf(num_joints_);
    for (int j = 0; j < num_joints_; j++) {
        conf[j] = trajectory_(0, j);
    }

    if(!iksolver_.collisionChecking(conf)){
        return true;
    }

    return collision;
}

bool RcikPlanning::jointVelocityLimitCheck(const KDL::JntArray &q) {
    if(!parameters_.use_checking_velocity_limit_){
        return true;
    }

    double vt = parameters_.time_duration_;

    for (uint j = 0; j < num_joints_; j++){
        double mt = std::abs(q(j) - trajectory_(1, j));
        if(mt > M_PI){
            mt = 2*M_PI - mt;
        }
        mt /= vt;
        if(mt > vel_limit_[j]){
            return false;
        }
    }

    return true;
}

bool RcikPlanning::kinematicSingularityCheck(const KDL::JntArray &q) {
    if(!parameters_.use_checking_singularity_){
        return true;
    }

    KDL::Jacobian jac(num_joints_);

    iksolver_.getJacobian(q, jac);
    double yosh = std::abs(std::sqrt((jac.data * jac.data.transpose()).determinant()));

    return yosh >= parameters_.min_singularity_value_;
}

void RcikPlanning::registerParents(const moveit::core::JointModel* model)
{
    const moveit::core::JointModel* parent_model = NULL;
    bool found_root = false;

    if (model == kmodel_->getRootJoint())
        return;

    while (!found_root)
    {
        if (parent_model == NULL)
        {
            if (model->getParentLinkModel() == NULL)
            {
                ROS_ERROR_STREAM("Model " << model->getName() << " not root but has NULL link model parent");
                return;
            }
            else if (model->getParentLinkModel()->getParentJointModel() == NULL)
            {
                ROS_ERROR_STREAM("Model " << model->getName() << " not root but has NULL joint model parent");
                return;
            }
            parent_model = model->getParentLinkModel()->getParentJointModel();
        }
        else
        {
            if (parent_model == kmodel_->getRootJoint())
            {
                found_root = true;
            }
            else
            {
                parent_model = parent_model->getParentLinkModel()->getParentJointModel();
            }
        }
        joint_parent_map_[model->getName()][parent_model->getName()] = true;
    }
}
