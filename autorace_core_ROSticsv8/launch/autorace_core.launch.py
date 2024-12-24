import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node
from launch.actions import TimerAction


def generate_launch_description():
    robot_control_node = Node(
        package='robot_comp_test',
        executable='ros_competition_node',
        name='robot_control_node',
        output='screen',
        parameters=[
            {"use_sim_time": True}
        ]
    )

    start_node = Node(
        package='robot_comp_test',
        executable='start_on_traffic_green_node',
        name='start_on_traffic_green_node',
        output='screen',
        parameters=[
            {"use_sim_time": True}
        ]
    )

    return LaunchDescription([
        
        robot_control_node,  # Добавляем узел управления роботом
        start_node
    ])