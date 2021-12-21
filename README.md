# spco2_boo
This repository is SpCoSLAM (added Bag-of-Objects) package.  
SpCoSLAM learns the relationship between object and place.  
Mainly, it is a program that executes [SpCoSLAM](https://github.com/a-taniguchi/SpCoSLAM2) in Gazebo.

*   Maintainer: Shoichi Hasegawa ([hasegawa.shoichi@em.ci.ritsumei.ac.jp](mailto:hasegawa.shoichi@em.ci.ritsumei.ac.jp)).
*   Author: Shoichi Hasegawa ([hasegawa.shoichi@em.ci.ritsumei.ac.jp](mailto:hasegawa.shoichi@em.ci.ritsumei.ac.jp)).

You can do it like this below image.  
(You need to prepare room layouts personally.)
![hsr-noetic](https://user-images.githubusercontent.com/74911522/137430543-1d35d631-963c-446e-ac13-560b64926d47.png)


## Content
* [Execution Environment](#execution-environment)
* [Execute Procedure](#execute-procedure)
* [Folder](#folder)
* [To Do](#to-do)
* [Reference](#reference)
* [Acknowledgements](#acknowledgements)


## Execution environment  
- Ubuntu：20.04LTS
- ROS：Noetic
- Python：3.8.10
- C++：14
- Robot：Human Support Robot (HSR)
- YOLOv5 (Emergent Systems Lab original)


## Execution Procedure
1  `cd HSR/catkin_ws/src`  
2. `git clone https://github.com/Shoichi-Hasegawa0628/spco2_boo.git`  
3. `cd ~/HSR/ && bash ./RUN-DOCKER-CONTAINER.bash`  
4. `cd catkin_ws`  
5. `catkin_make`   
6. Launch Gazebo  
7. Launch Rviz  
8. Launch YOLOv5
9. `rosnode kill /pose_integrator`   
10. `roscd rgiro_spco2_slam`  
11. `cd bash`  
12. `bash reset-spco2-slam-output.bash`  
13. `roslaunch rgiro_spco2_slam spco2_slam.launch`  
14. `roslaunch rgiro_spco2_slam spco2_word.launch`  

Teaching the place name while teleoping with `rqt`.

## Folder
- `rgiro_openslam_gmapping`：SpCoSLAM Wrapper of FastSLAM2.0 published on [OpenSLAM](https://openslam-org.github.io/)
- `rgiro_slam_gmapping`：SpCoSLAM Wrapper of slam_gmapping ros package (ros wrapper of openslam_gmapping)
- `rgiro_spco2_slam`：Main codes of SpCoSLAM
- `rgiro_spco2_visualization`：Visualization codes of learning spatial concepts

## To Do (Japanese)
- 絶対パスが含まれている箇所があるため、変更が必要な場合あり


## Reference
- [SpCoSLAM 2.0](https://github.com/a-taniguchi/SpCoSLAM2)
