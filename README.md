# Multimodal data collection pipeline on Panda

Collect 6d pose displacement with a Franka Panda robot arm.

## Scripts

To run data collection, do the followings:

1. Start the panda controller.
   1. Unlock joint, open the safety switch, enter FCI mode.
   2. Run the following command
```sh
electro # this macro sets up the correct environmental variables for ROS and Franka
cd ~/src/frankapy
./bash_scripts/start_control_pc.sh -u gelsight -i gelsight-panda -d src/franka-interface -g 0
``` 

2. Start data collection. Open another terminal, then:
```sh
electro # this macro sets up the correct environmental variables for ROS and Franka
cd ~/src/FoundationTactile/TheProbe/ProbingPanda
python scripts/disp_collection.py object_overwrite=<object_name> env.init_pose_cnt=<traj_cnt>
```

Here `<object_name>` is a string wrapped inside double quote specifiying which object we are probing now.
If it has multiple words, the words should be seperated by `_` (underscore).
Example: <object_name> = `"black_clamp"`

`<traj_cnt>` should be an integer specifying the trajectory count for this particular object.

---

After the above script starts running, the followings are what happen next:
1. It first homes the robot.
2. It shows a screen of what the tactile sensors can see. The original purpose is to use this to confirm we are using the correct ordering of the two sensors. But this is not needed anymore. Press `q` on the visualization window to quit the visualization.
3. It goes to a pre-grasp pose, then enters guide mode for 10s, during which please grab the robot EE and move it to a good initial position. At the end of the 10s, the script will prompt you whether you have finished adjusting the init pose. Press `n` then `Enter` if you need another 10s, or just press `Enter` if it's good to go.
4. One trial takes about 30min to finish. After it finishes, take another object, or choose another init pose.
   
