""" mouse model example in pybullet. """

import os
import time

import farms_pylog as pylog
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data

from bullet_simulation import BulletSimulation
from farms_container import Container
#from farms_core.units import SimulationUnitScaling

pylog.set_level('error')


class MouseSimulation(BulletSimulation):
    """ Mouse Simulation Class """

    def __init__(self, container, sim_options):
        super(MouseSimulation, self).__init__(
            #container, SimulationUnitScaling(), **sim_options
            container, **sim_options
        )
        self.connection_mode = p.getConnectionInfo(
            self.physics_id
        )['connectionMethod']
        if self.MUSCLES and (self.connection_mode == 1):
            u = container.muscles.activations
            self.muscle_params = {}
            self.muscle_excitation = {}
            for muscle in self.muscles.muscles.keys():
                self.muscle_params[muscle] = u.get_parameter(
                    'stim_{}'.format(muscle)
                )
                self.muscle_excitation[muscle] = p.addUserDebugParameter(
                    "flexor {}".format(muscle), 0, 1, 0.00
                )
                self.muscle_params[muscle].value = 0
        else:
            self.torque = p.addUserDebugParameter(
                    "torque", -0.5, 0.5, 0.00
                )


    def controller_to_actuator(self):
        """ Implementation of abstractmethod. """
        if self.TIME > 1.0:
                self.container.muscles.activations.set_parameter_value("stim_RIGHT_HIND_AB", 1.0)
        else:
            self.container.muscles.activations.set_parameter_value("stim_RIGHT_HIND_AB", 0.0)
        if self.MUSCLES and (self.connection_mode == 1):
            # for muscle in self.muscles.muscles.keys():
            #     self.muscle_params[muscle].value = p.readUserDebugParameter(
            #         self.muscle_excitation[muscle]
            #     )
            pass
            if self.TIME > 1.0:
                self.container.muscles.activations.set_parameter_value("stim_RIGHT_HIND_AB", 1.0)
        else:
            # p.setJointMotorControl2(
            #     self.animal,
            #     self.joint_id["LWrist_flexion"],
            #     p.TORQUE_CONTROL,
            #     force=p.readUserDebugParameter(self.torque)*1e-2
            # )
            pass
        # self.muscle_params["RIGHT_HIND_SM"].value = np.sin(
        #     2*np.pi*self.TIME
        # )
        # self.muscle_params["RIGHT_HIND_RF"].value = np.sin(
        #     2*np.pi*self.TIME+np.pi
        # )
        # self.muscle_params["RIGHT_HIND_POP"].value = np.sin(
        #     2*np.pi*self.TIME
        # )
        # self.muscle_params["RIGHT_HIND_OE"].value = np.sin(
        #     2*np.pi*self.TIME
        # )
        # self.muscle_params["RIGHT_HIND_CF"].value = np.sin(
        #     2*np.pi*self.TIME+np.pi
        # )
        # self.muscle_params["RIGHT_HIND_GM_mid"].value = 0.5
        # self.muscle_params["RIGHT_HIND_GM_dorsal"].value = 0.5
        # self.muscle_params["RIGHT_HIND_GM_ventral"].value = 0.5

    def feedback_to_controller(self):
        """ Implementation of abstractmethod. """
        pass

    def update_parameters(self):
        """ Implementation of abstractmethod. """
        pass

    def optimization_check(self):
        """ Implementation of abstractmethod. """
        pass


def main():
    """ Main """

    sim_options = {
        "headless": False,
        #"model": "../../data/models/sdf/right_hindlimb.sdf",
         "model": "../../data/models/sdf/mouse_with_joint_limits.sdf",
        "model_offset": [0., 0., 0.075],
        "floor_offset": [0, 0, 0],
        "run_time": 2.5,
        "planar": False,
        "muscles": "../../data/config/muscles/right_forelimb.yaml",
        # "muscles" : "../../data/config/temp_estimated_left_forelimb_muscles.yaml",
        "track": False,
        "camera_yaw": -270,
        "camera_distance": 0.075,
        "slow_down": True,
        "sleep_time": 1e-3,
        "record" : False,
        "pose": "../../data/config/locomotion_pose.yaml"
        # "base_link": "Pelvis"
    }
    container = Container(
        max_iterations=int(sim_options['run_time']/0.001)
    )
    animal = MouseSimulation(container, sim_options)
    animal.run()
    container.dump(overwrite=True)


if __name__ == '__main__':
    main()
