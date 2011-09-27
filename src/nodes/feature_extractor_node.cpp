#include <ros/ros.h>
#include <nodelet/loader.h>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "feature_extractor", ros::init_options::AnonymousName);
  if (ros::names::remap("image") == "image") {
    ROS_WARN("Topic 'image' has not been remapped!");
  }

  ROS_INFO("Creating manager...");
  nodelet::Loader manager(false);
  ROS_INFO("Manager created...");
  nodelet::M_string remappings;
  nodelet::V_string my_argv(argv + 1, argv + argc);

  ROS_INFO("Loading nodelet as %s...", ros::this_node::getName().c_str());
  bool success = manager.load(ros::this_node::getName(), "stereo_feature_extraction/feature_extractor", remappings, my_argv);
  if (success)
  {
    ROS_INFO("Nodelet loaded.");
  }
  else
  {
    ROS_ERROR("Failed to load nodelet!");
  }


  ros::spin();
  return 0;
}
