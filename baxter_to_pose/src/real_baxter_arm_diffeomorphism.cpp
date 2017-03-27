#include <iostream>
#include <string>
#include <cmath>
#include <math.h>
#include <fstream>
#include <sstream>
#include <boost/algorithm/string.hpp>
#include "../cppDiffeo/diffeoMovements.hpp"
#include "../cppDiffeo/diffeoSearch.hpp"
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <tf/transform_listener.h>
#include <geometry_msgs/PoseStamped.h>
#include <baxter_core_msgs/EndEffectorCommand.h>
#include <baxter_core_msgs/SolvePositionIK.h>
#include <baxter_core_msgs/JointCommand.h>
//#include <vector>


using namespace std;
using namespace Eigen;
using namespace DiffeoMethods;
using namespace DiffeoMovements;

std::vector<double> left_joints_values(7);
std::vector<double> left_joints_velocities(7);
std::vector<double> desired_velocity(7);
std::vector<double> new_pose(7);

int sgn(double d){
    if(d<0)
        return -1;
    else if (d>0)
        return 1;
    else
        return 0;
}

void jocommCallback(sensor_msgs::JointState jo_state)
{
    //all_joint_names = jo_state.name;
    left_joints_values[0] = jo_state.position[4];
    left_joints_values[1] = jo_state.position[5];
    left_joints_values[2] = jo_state.position[2];
    left_joints_values[3] = jo_state.position[3];
    left_joints_values[4] = jo_state.position[6];
    left_joints_values[5] = jo_state.position[7];
    left_joints_values[6] = jo_state.position[8];
    left_joints_velocities[0] = jo_state.velocity[4];
    left_joints_velocities[1] = jo_state.velocity[5];
    left_joints_velocities[2] = jo_state.velocity[2];
    left_joints_velocities[3] = jo_state.velocity[3];
    left_joints_velocities[4] = jo_state.velocity[6];
    left_joints_velocities[5] = jo_state.velocity[7];
    left_joints_velocities[6] = jo_state.velocity[8];
}

//several diffeo in velocity
int main(int argc, char **argv){
    ros::init(argc, argv, "arm_control");
    //Get some parameters
    diffeoSearchOpts theseOptions = diffeoSearchOpts(80);
    //Change the scaling
    theseOptions.thisScalingMethod = maximal;

    //initiate everything needed thing for baxter
    ros::AsyncSpinner spinner (1);
    spinner.start();
    ros::NodeHandle node;
    baxter_core_msgs::JointCommand command_msg;
    ros::Subscriber sub_jointmsg = node.subscribe<sensor_msgs::JointState>("/robot/joint_states",1,jocommCallback);
    ros::Publisher gripper_pub = node.advertise<baxter_core_msgs::EndEffectorCommand>("/robot/end_effector/left_gripper/command", 1);
    ros::Publisher pub_msg=node.advertise<baxter_core_msgs::JointCommand>("/robot/limb/left/joint_command",1);
    ros::Rate rate(1000.0);
    ros::AsyncSpinner my_spinner(1);
    my_spinner.start();

    baxter_core_msgs::EndEffectorCommand gripper_command;
    gripper_command.id = 65538;
    gripper_command.command = "calibrate";
    gripper_pub.publish(gripper_command);

    std::ifstream BigFile("/home/qcensier/catkin_ws/src/baxter_to_pose/taking_cube_to_column/the_move");
    std::ofstream Wfile("./DiffeoTrajectory");
    std::string line;
    std::getline(BigFile,line);
    int num_diffeo=stoi(line);

    if(Wfile.is_open()){

        //timer here

        for (int i=1; i<=num_diffeo;i++){
            std::getline(BigFile,line);
            int close_gripper=stoi(line);
            std::cout << close_gripper << std::endl;

            //std::vector<double> gripper={0.,0.};
            //if (close_gripper==1){
            //////////////////////note that here we need to know how to move the gripper
            //    gripper={0.32,-0.32};
            //}
            //else{
            //    gripper={0.,0.};
            //}

            std::vector<double> starting_angles=left_joints_values;

            std::getline(BigFile,line);
            string diffeoPath = line;
            std::cout << diffeoPath << std::endl;

            MatrixXd resPos(7,1);
            MatrixXd resVel(7,1);
            VectorXd actPos(7,1);

            //////////////////////////////////////////////////////////
            double norm=1;

            MatrixXd targetMat = Leph::ReadMatrix(diffeoPath);
            //targetMat = targetMat.leftCols(targetMat.cols()-1);
            //targetMat = targetMat.transpose();
            int col=targetMat.cols();
            VectorXd timeVec(col);
            for (int i=0; i<col;i++){
                timeVec(i)=0.05*(col-1-i);
            }
            //Prepare results
            diffeoStruct thisDiffeo;
            diffeoDetails thisDetails;
            searchDiffeoMovement( thisDiffeo, thisDetails, targetMat, timeVec,  "" , theseOptions );
            DiffeoMoveObj thisMovement;
            thisMovement.setDiffeoStruct(thisDiffeo);
            thisMovement.setDiffeoDetailsStruct(thisDetails);
            VectorXd tStepsVec(1);
            tStepsVec(0)=0.050;

            //creating an eigen values matrix to converge faster to the trajectory than to the final point
            Eigen::MatrixXd MM=-MatrixXd::Identity(7,7);
            MM=MM*4;
            MM(0,0)=-1.0;

            thisMovement.doInit();
            thisMovement.setZoneAi(0,MM);
            thisMovement.setZoneFunction(ret0());
            thisMovement.setScaleFunc(standardScale());

            double begin=ros::Time::now().toSec();

            if (close_gripper==1){
                gripper_command.command = "grip";
                gripper_pub.publish(gripper_command);
                std::cout << "closing"<<std::endl;
                begin=ros::Time::now().toSec();
                while(ros::Time::now().toSec()-begin < 1.5){}
            }


            //double dt=0.1;

            while(ros::Time::now().toSec()-begin < 10 && norm>0.07){
                starting_angles=left_joints_values;
               if(starting_angles[0]*starting_angles[0]>0.00001){
                    for (int i=0; i<7;i++){
                        actPos(i)=starting_angles[i];
                    }
                    thisMovement.getTraj(resPos, resVel, actPos, tStepsVec);
                    std::vector<double> actual_velocity=left_joints_velocities;
                    double alpha=1;
                    for (int i=0; i<7;i++){
                        double vel=resVel(i);
                        desired_velocity[i]=actual_velocity[i] + alpha*(0.30*vel-actual_velocity[i]);
                    }

                    command_msg.mode=command_msg.VELOCITY_MODE;
                    command_msg.names={"left_s0", "left_s1","left_e0", "left_e1", "left_w0", "left_w1", "left_w2"};
                    command_msg.command=desired_velocity;
                    //double tnow=ros::Time::now().toSec();
                   // while(ros::Time::now().toSec()-tnow<1.0)
                    pub_msg.publish(command_msg);

                    starting_angles=left_joints_values;
                    norm=0;
                    for(int i = 0; i < 7; i++){
                        Wfile << starting_angles[i] << " ";
                        norm=norm+(resPos(i)-starting_angles[i])*(resPos(i)-starting_angles[i]);
                    }
                    Wfile << ros::Time::now().toSec()-begin << "\n";
                }
                //rate.sleep();
            }
           // std::cout << begin << std::endl;

            gripper_command.command = "release";
            gripper_pub.publish(gripper_command);
            std::cout << "opening" <<std::endl;

            command_msg.mode=command_msg.VELOCITY_MODE;
            command_msg.names={"left_s0", "left_s1","left_e0", "left_e1", "left_w0", "left_w1", "left_w2"};
            command_msg.command={0.0,0.0,0.0,0.0,0.0,0.0,0.0};
            pub_msg.publish(command_msg);

           if (close_gripper!=1){
                gripper_command.command = "grip";
                gripper_pub.publish(gripper_command);
                std::cout << "closing"<<std::endl;
            }
            else{
                gripper_command.command = "release";
                gripper_pub.publish(gripper_command);
                std::cout << "opening" <<std::endl;
            }
            begin=ros::Time::now().toSec();
        }
    }
    else{
        std::cerr << "impossible to open " << argv[1] << std::endl;
        exit(1);
    }
    return 0;
}
