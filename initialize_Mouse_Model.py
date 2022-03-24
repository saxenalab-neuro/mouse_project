import pybullet as p
import time
import pybullet_data
import yaml
import numpy as np

physicsClient = p.connect(p.GUI) #or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,0) #no gravity

p.setPhysicsEngineParameter(fixedTimeStep= .25, 
            numSolverIterations=100, 
            enableFileCaching=0, 
            numSubSteps=1,
            solverResidualThreshold=1e-10,
            #erp=0.0,
            contactERP=0.0, frictionERP=0.0,
            reportSolverAnalytics = 1)

planeId = p.loadURDF("plane.urdf")

mouseId = p.loadSDF("/files/mouse_with_joint_limits.sdf", globalScaling = 25)[0] #if all files are downloaded, use relative path
#mouseId = p.loadSDF("/Users/andreachacon/mouse_biomechanics_paper/data/models/sdf/right_forelimb.sdf", globalScaling = 100)[0] # only arm
num_joints = p.getNumJoints(mouseId) #225
model_offset = (0.0, 0.0, 1.2) #z position modified with global scaling
#model_offset = (-.7, -6,4.4) # for arm only
pose_file = "files/default_pose.yaml" 

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

def initialize_position(pose_file, joint_list):
    with open(pose_file) as stream:
        data = yaml.load(stream, Loader=yaml.SafeLoader)
        data = {k.lower(): v for k, v in data.items()}
    for joint in joint_list:
        joint_name =p.getJointInfo(mouseId, joint)[1] 
        _pose = np.deg2rad(data.get(p.getJointInfo(mouseId, joint)[1].decode('UTF-8').lower(), 0))#decode removes b' prefix
        p.resetJointState(mouseId, joint, targetValue=_pose)

#jointId_dict = generate_name_to_joint_id_dict(mouseId)
#joint_list = initialize_joint_list(num_joints)
#joint_dictionary = generate_joint_id_to_name_dict(mouseId)
p.resetBasePositionAndOrientation(mouseId, model_offset, p.getQuaternionFromEuler([0, 0, 80.2]))
#initialize_position(pose_file, joint_list)

p.setJointMotorControlArray(
            mouseId,
            np.arange(num_joints),
            p.VELOCITY_CONTROL,
            targetVelocities=np.zeros((num_joints,)),
            forces=np.zeros((num_joints,))
        )
#the following should be moved to a main/simulation file

#print(p.getJointInfo(mouseId, 107))
#p.enableJointForceTorqueSensor(mouseId, 106, 1)
for i in range (10000):
    #posObj = p.getJointState(mouseId, 106) #not working
    #print(posObj)
    #p.setJointMotorControl2(mouseId, 105, p.TORQUE_CONTROL, force = .002)
    #forces = [.25, .25, .25, .25, .25, .25, .25]
    #p.setJointMotorControlArray(mouseId, arm_indexes, p.TORQUE_CONTROL, forces = forces)
    p.stepSimulation()
    time.sleep(.001)
p.disconnect()
