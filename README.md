# RoboticsCompetition

## Dependencies
```
sudo apt install ros-humble-tf-transformations
pip3 install numpy==1.26
pip3 install transforms3d --upgrade
```

## To launch:
Terminal 1:

```
ros2 launch robot_bringup autorace_2023.launch.py
```

Terminal 2:

```
ros2 launch autorace_core_ROSticsv8 autorace_core.launch.py
```

Terminal 3:

```
ros2 run referee_console mission_autorace_2023_referee
```
