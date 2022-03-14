# RCIK

1. Install the dependencies.

<pre> MoveIt!, Trac-ik, Torch </pre>

2. Download the learned model at https://sglab.kaist.ac.kr/RCIK
  - Move the model to "/src/data/model" folder

3. Setup your robot.

<pre> roslaunch moveit_setup_assistant setup_assistant.launch </pre>
  Make your PLANNING_GROUP

4. Set your "Torch" path at CMakeLists.txt.

5. Download the trained model via ~.

6. Execute the code.
  - Run the "move_group.launch" for your robot.
  - Run the problem using a launch file. (You can change a scene through the launch file.)
    <pre> roslaunch rcik fetch_keyboard.launch </pre>
  - Run the planning code.
    <pre> rosrun rcik keyboard </pre>
    - You should change the "main.cpp" according to your settings.

