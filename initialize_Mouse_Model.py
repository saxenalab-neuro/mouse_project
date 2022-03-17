import pybullet as p
import time
import pybullet_data
import yaml
import numpy as np

physicsClient = p.connect(p.GUI) #or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,0) #no gravity

p.setPhysicsEngineParameter(
            fixedTimeStep= .25,
            numSolverIterations=100,
            enableFileCaching=0,
            numSubSteps=1,
            solverResidualThreshold=1e-10,
            # erp=0.0,
            contactERP=0.0,
            frictionERP=0.0,
        )

planeId = p.loadURDF("plane.urdf")

boxId = p.loadSDF("/Users/andreachacon/mouse_biomechanics_paper/data/models/sdf/mouse.sdf", globalScaling = 25) #modify absolute path
mouseId = boxId[0]
num_joints = p.getNumJoints(mouseId) #225
model_offset = (0.0, 0.0, 1.2) #z position modified with global scaling
pose_file = "/Users/andreachacon/mouse_biomechanics_paper/data/config/hind_limb_default_pose_bullet.yaml" #modify absolute path

def initialize_joint_list(num_joints):
    joint_list =[]
    for joint in range(225):
        joint_list.append(joint)
    return joint_list
    #RShoulder_rotation - 104
    #RShoulder_adduction - 105
    #RShoulder_flexion - 106
    #RElbow_flexion - 107
    #RElbow_supination - 108
    #RWrist_adduction - 110
    #RWrist_flexion - 111

arm_indexes = [104, 105, 106, 107, 108, 110, 111]


def generate_joint_id_to_name_dict(mouseId):
    joint_Dictionary ={}
    for i in range(p.getNumJoints(mouseId)):
        joint_Dictionary[i] = p.getJointInfo(mouseId, i)[1].decode('UTF-8') 
    return joint_Dictionary

def generate_name_to_joint_id_dict(mouseId):
    name_Dictionary ={}
    for i in range(p.getNumJoints(mouseId)):
        name_Dictionary[p.getJointInfo(mouseId, i)[1].decode('UTF-8')] = i
    return name_Dictionary

jointId_dict = generate_name_to_joint_id_dict(mouseId)
print(jointId_dict)

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

joint_list = initialize_joint_list(num_joints)
#print(joint_list)
joint_dictionary = generate_joint_id_to_name_dict(mouseId)
p.resetBasePositionAndOrientation(mouseId, model_offset, p.getQuaternionFromEuler([0., 0., 80]))
initialize_position(pose_file, joint_list)

#the following should be moved to a main/simulation file
p.setRealTimeSimulation(0)
p.enableJointForceTorqueSensor(mouseId, 104, 1)
for i in range (10000):
    #posObj = p.getJointState(mouseId, 104)
    #print(posObj)
    #p.setJointMotorControl2(mouseId, 104, p.TORQUE_CONTROL, (.25), force = 2)
    #forces = [.25, .25, .25, .25, .25, .25, .25]
    #p.setJointMotorControlArray(mouseId, arm_indexes, p.TORQUE_CONTROL, forces = forces)
    p.stepSimulation()
    time.sleep(.001)
p.disconnect()
