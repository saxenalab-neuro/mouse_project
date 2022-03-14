import pybullet as p
import time
import pybullet_data
import yaml
import numpy as np

physicsClient = p.connect(p.GUI) #or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,0) #no gravity
planeId = p.loadURDF("plane.urdf")

boxId = p.loadSDF("/Users/andreachacon/mouse_biomechanics_paper/data/models/sdf/mouse_with_joint_limits.sdf", globalScaling = 25) #modify absolute path
mouseId = boxId[0]
num_joints = p.getNumJoints(mouseId) #225
#print('num joints', num_joints)
model_offset = (0.0, 0.0, 1.2) #z position modified with global scaling
joint_list = []
pose_file = "/Users/andreachacon/mouse_biomechanics_paper/data/config/default_pose.yaml" #modify absolute path

def initialize_joint_list(mouseId, num_joints):
    for joint in range(num_joints):
        joint_list.append(joint)
        #print (joint, p.getJointInfo(mouseId, joint)[1])
    return joint_list
    #RShoulder_rotation - 104
    #RShoulder_adduction - 105
    #RShoulder_flexion - 106
    #RElbow_flexion - 107
    #RElbow_supination - 108
    #RWrist_adduction - 110
    #RWrist_flexion - 111

def initialize_position(pose_file, joint_list):
    with open(pose_file) as stream:
        data = yaml.load(stream, Loader=yaml.SafeLoader)
        data = {k.lower(): v for k, v in data.items()}
    #print(data)
    for joint in joint_list:
        #print(data.get(p.getJointInfo(mouseId, joint)[1]))
        joint_name =p.getJointInfo(mouseId, joint)[1] 
        _pose = np.deg2rad(data.get(p.getJointInfo(mouseId, joint)[1].decode('UTF-8').lower(), 0))#decode removes b' prefix
        #print(p.getJointInfo(mouseId, joint)[1].decode('UTF-8').lower(), _pose)
        p.resetJointState(mouseId, joint, targetValue=_pose)

joint_list = initialize_joint_list(mouseId, num_joints)
p.resetBasePositionAndOrientation(mouseId, model_offset, p.getQuaternionFromEuler([0., 0., 80]))
#initialize_position(pose_file, joint_list)

#the following will be moved to a main file
p.setRealTimeSimulation(0)
for i in range (10000):
    posObj = p.getJointState(mouseId, 0)[0]
    print(posObj)
    #p.setJointMotorControl2(mouseId, 104, p.POSITION_CONTROL, (posObj + np.pi/4))
    p.stepSimulation()
    #time.sleep(1./240.)
#cubePos, cubeOrn = p.getBasePositionAndOrientation(mouseId)
#print(cubePos,cubeOrn)
p.disconnect()