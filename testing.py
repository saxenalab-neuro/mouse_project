import pybullet as p
import pybullet_data

import scripts.model_utils as model_utils

file_path = "/Users/andreachacon/Documents/GitHub/mouse_project/files/mouse_test.sdf" 
pose_file = "/Users/andreachacon/Documents/GitHub/mouse_project/files/default_pose.yaml"

model_offset = (0.0, 0.0, 1.2) #z position modified with global scaling

client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.8) #normal gravity
plane = p.loadURDF("plane.urdf") #sets floor



#unstable
#block = p.loadURDF("block.urdf", globalScaling = 6)
#p.resetBasePositionAndOrientation(block, (.77, .3, 1), p.getQuaternionFromEuler([0, 7.88, 4.5]))
#block2 = p.loadURDF("block.urdf", globalScaling = 6)
#p.resetBasePositionAndOrientation(block2, (.77, -.3, 1), p.getQuaternionFromEuler([0, 7.88, 4.5]))

#too short
#table = p.loadURDF("table/table.urdf")
#p.resetBasePositionAndOrientation(table, (.7, 0, 0), p.getQuaternionFromEuler([0, 0, 4.75]))

model = p.loadSDF(file_path, globalScaling = 25)[0]#resizes, loads model, returns model id
p.resetBasePositionAndOrientation(model, model_offset, p.getQuaternionFromEuler([0, 0, 80.2]))

#model_utils.reset_model_position(model, pose_file)
print(model_utils.generate_name_to_joint_id_dict(model))

for i in range (100000):
    p.stepSimulation()
    #p.resetBasePositionAndOrientation(block, (.77, 0.3, .56), p.getQuaternionFromEuler([0, 7.88, 4.5]))
    #p.resetBasePositionAndOrientation(block2, (.77, -.3, .56), p.getQuaternionFromEuler([0, 7.88, 4.5]))
    #p.resetBasePositionAndOrientation(table, (.7, 0, 0), p.getQuaternionFromEuler([0, 0, 4.75])) 

    
p.disconnect()