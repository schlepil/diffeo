#include <ros/ros.h>
#include <baxter_core_msgs/EndEffectorCommand.h>


int main(int argc, char** argv)
{
    ros::init(argc, argv, "gripper_control");
    ros::NodeHandle nh;
    ros::Publisher gripper_pub = nh.advertise<baxter_core_msgs::EndEffectorCommand>("/robot/end_effector/left_gripper/command", 1);


    ros::AsyncSpinner my_spinner(1);
    my_spinner.start();

    baxter_core_msgs::EndEffectorCommand command;
    command.id = 65538;
    while(ros::ok()){
        //for calibrating the gripper u need the gripper id and the command "calibrate"


        command.command = "calibrate";
        gripper_pub.publish(command);
    }

    while(ros::ok()){
        //close gripper command "grip"
        //command.command = "grip";
        command.command = "release";

        gripper_pub.publish(command);
    }


    /*
    //open gripper command "release"
    command.command = "release";
    gripper_pub.publish(command);*/
    return 0;
}
