import os
import sys
import time
import numpy as np

sys.path.append(os.path.abspath("./gym_dcmm"))

import mujoco
import mujoco.viewer

from configs.env import DcmmCfg
from gym_dcmm.agents.MujocoDcmm import xml_to_string
from robot_info import load_model_and_data
from gym_dcmm.utils.ik_pkg.ik_arm import IKArm
from gym_dcmm.utils.util import calculate_arm_Te
from gym_dcmm.utils.pid import PID


def ik_arm_solve(model, data, ik_solver, target_pos, target_quat):
    """
    Solves inverse kinematics for the arm to reach the target pose.
    """
    Tep = calculate_arm_Te(target_pos, target_quat)
    q0_full = np.zeros(model.nv)
    q0_full[0:6] = data.qpos[0:6]  # arm joints are 0–5
    result = ik_solver.solve(model, data, Tep, q0_full)
    return result


def dummy_hand_controller(t):
    """
    Dummy controller for the hand.
    Returns target joint positions for 16 hand joints.
    """
    base = np.array(DcmmCfg.hand_joints)
    oscillation = 0.1 * np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz oscillation
    return base + oscillation


def dummy_object_controller(t):
    """
    Dummy controller for the object actuators.
    Returns control values for 8 object actuators.
    """
    return 0.05 * np.sin(2 * np.pi * 0.2 * t) * np.ones(8)  # 0.2 Hz


def main():
    model_path = "urdf/xarm_leap_with_objects.xml"
    model, data = load_model_and_data(model_path)
    print(f"Model loaded from {os.path.abspath(model_path)}")

    ik_solver = IKArm()
    arm_pid = PID(
        "arm", 
        DcmmCfg.Kp_arm, 
        DcmmCfg.Ki_arm, 
        DcmmCfg.Kd_arm, 
        dim=6, 
        llim=DcmmCfg.llim_arm, 
        ulim=DcmmCfg.ulim_arm, 
        debug=False
    )
    hand_pid = PID(
        "hand", 
        DcmmCfg.Kp_hand, 
        DcmmCfg.Ki_hand, 
        DcmmCfg.Kd_hand, dim=16, 
        llim=DcmmCfg.llim_hand, 
        ulim=DcmmCfg.ulim_hand, 
        debug=False
    )

    target_pos = np.array([0.5, 0.0, 0.2])
    target_quat = np.array([0.0, 0.0, 0.0, 0.0])

    result = ik_arm_solve(model, data, ik_solver, target_pos, target_quat)
    print("IK result:", result)
    target_arm_qpos, success = result[0], result[1]

    if not success:
        print("IK solving failed!")
        return
    print("IK success! Target joint positions:", target_arm_qpos)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer launched. Close the window to stop.")

        while viewer.is_running():
            t = data.time

            current_arm_qpos = data.qpos[0:6]
            current_hand_qpos = data.qpos[6:22]

            mv_arm = arm_pid.update(target_arm_qpos[0:6], current_arm_qpos, t)
            target_hand_qpos = dummy_hand_controller(t)
            mv_hand = hand_pid.update(target_hand_qpos, current_hand_qpos, t)
            mv_object = dummy_object_controller(t)

            # Combine all control signals
            ctrl = np.zeros(model.nu)
            ctrl[0:6] = mv_arm
            ctrl[6:22] = mv_hand
            ctrl[22:30] = mv_object

            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
