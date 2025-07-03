import sys
import os
import numpy as np
sys.path.append(os.path.abspath("./gym_dcmm"))
import mujoco
import mujoco.viewer
from configs.env import DcmmCfg
from gym_dcmm.agents.MujocoDcmm import xml_to_string

XML_XARM_LEAP_WITH_OBJECT_PATH = "urdf/xarm_leap_with_objects.xml"
XML_XARM_LEAP_PATH = "urdf/xarm_leap.xml"
XML_X1_XARM6_LEAP_RIGHT_OBJECT_PATH = "urdf/x1_xarm6_leap_right_object.xml"


def load_model_and_data(xml_rel_path):
    """Load MuJoCo model and data from xml path."""
    model_path = os.path.join(DcmmCfg.ASSET_PATH, xml_rel_path)
    print(f"Model loaded from {model_path}")
    xml_string = xml_to_string(model_path)
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    return model, data

def print_joint_info(model):
    """Print joint index, name and qpos address."""
    print("Joint names in order:")
    for i in range(model.njnt):
        name = model.joint(i).name
        addr = model.jnt_qposadr[i]
        print(f"{i}: {name} (qpos addr: {addr})")
    print(f"model.nq: {model.nq}, model.njnt: {model.njnt}")

def print_joint_stiffness_damping(model):
    """Print each joint's damping and linked actuator stiffness (if available)."""
    print(f"{'JointID':>7} | {'JointName':<25} | {'Damping':>10} | {'Stiffness (from actuator)':>25}")
    print("-" * 75)
    for j in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        damping = model.dof_damping[j]
        stiffness = None

        # Try to find actuator linked to this joint
        for a in range(model.nu):
            if model.actuator_trnid[a][0] == j:  # actuator affects this joint
                # Try gainprm[0] or biasprm[1] as stiffness proxy
                stiffness = model.actuator_gainprm[a][0]
                if stiffness == 0.0:
                    stiffness = model.actuator_biasprm[a][1]
                break  # Take first matching actuator

        print(f"{j:7} | {joint_name:<25} | {damping:10.4f} | {str(stiffness) if stiffness is not None else 'N/A':>25}")

def print_collisions(model, data):
    """Print current contact pairs (geom name or ID)."""
    if data.ncon == 0:
        print("No collisions detected.")
    else:
        for i in range(data.ncon):
            contact = data.contact[i]
            geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            print(f"Collision: {geom1 or contact.geom1} <--> {geom2 or contact.geom2}")


def debug_geom_mapping(model):
    """Print mapping from geom ID to geom name, body name, and joint name (if any)."""
    print(f"{'GeomID':>6} | {'GeomName':<25} | {'BodyName':<25} | {'JointName':<25}")
    print("-" * 90)
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        body_id = model.geom_bodyid[geom_id]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)

        # Try to find joint attached to same body
        joint_name = None
        for j in range(model.njnt):
            if model.jnt_bodyid[j] == body_id:
                joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                break

        print(f"{geom_id:6} | {str(geom_name):<25} | {str(body_name):<25} | {str(joint_name):<25}")

def debug_check_dof_vs_qpos(model, data):
    """
    Check the relationship between MuJoCo model DoFs (nv) and position variables (qpos),
    and identify which joint(s) cause nq > nv by contributing more qpos than DoFs.

    Args:
        model: The MuJoCo model object.
        data: The MuJoCo data object.
    """
    print("\n======= DOF vs QPOS Check =======")
    print(f"model.nv (number of DoFs)       = {model.nv}")
    print(f"model.nq (number of qpos entries) = {model.nq}")
    print(f"data.qpos.shape                 = {data.qpos.shape}")

    # Track which qpos entries are used
    qpos_has_dof = np.zeros(model.nq, dtype=bool)

    # Track detailed joint qpos/dof contributions
    total_qpos_from_joints = 0
    total_dofs_from_joints = 0

    print(f"\n{'JointID':>7} | {'Name':<25} | {'Type':<6} | {'qpos addr':>10} | {'#qpos':>6} | {'#DoF':>5}")
    print("-" * 70)

    for j in range(model.njnt):
        name = model.joint(j).name
        joint_type = model.jnt_type[j]
        qadr = model.jnt_qposadr[j]

        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            nq = 7
            nv = 6
            type_name = "FREE"
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            nq = 4
            nv = 3
            type_name = "BALL"
        else:
            nq = 1
            nv = 1
            type_name = "HINGE" if joint_type == mujoco.mjtJoint.mjJNT_HINGE else "SLIDE"

        total_qpos_from_joints += nq
        total_dofs_from_joints += nv
        qpos_has_dof[qadr:qadr+nq] = True

        print(f"{j:7} | {name:<25} | {type_name:<6} | {qadr:10} | {nq:6} | {nv:5}")

    # Check for unused qpos entries
    print("\n== QPOS entries with NO corresponding DOF:")
    unused_qpos = False
    for i in range(model.nq):
        if not qpos_has_dof[i]:
            print(f"qpos[{i}] has NO corresponding DOF")
            unused_qpos = True
    if not unused_qpos:
        print("All qpos entries are associated with some joint type.")

    # Summary
    print("\n======= Summary =======")
    print(f"Total qpos used by joints = {total_qpos_from_joints}")
    print(f"Total DoFs from joints    = {total_dofs_from_joints}")
    diff = model.nq - model.nv
    if diff > 0:
        print(f"Extra {diff} qpos entries exist due to joint types like FREE or BALL.")
    else:
        print("qpos and DoFs are consistent.")



def set_initial_pose(data):
    """Set initial pose for xArm and object."""
    # Joint init
    qpos = [0.0] * model.nq
    qpos[0] = 0.0
    qpos[1] = 0.0
    qpos[2] = 0.0
    qpos[3] = 1.5
    qpos[4] = 3.14
    qpos[5] = 1.57
    qpos[6] = 0.0

    # Object pose
    qpos[23:26] = [0.55, 0.0, 0.90]
    qpos[26:30] = [1.0, 0.0, 0.0, 0.0]

    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

if __name__ == "__main__":
    model, data = load_model_and_data(XML_XARM_LEAP_WITH_OBJECT_PATH)
    # model, data = load_model_and_data(XML_XARM_LEAP_PATH)
    # model, data = load_model_and_data(XML_X1_XARM6_LEAP_RIGHT_OBJECT_PATH)
    
    # print_joint_info(model)
    # debug_geom_mapping(model) 
    # print_collisions(model, data)
    # print_joint_stiffness_damping(model)
    debug_check_dof_vs_qpos(model, data)
    
    # set_initial_pose(data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer launched. Close the window to stop.")
        while viewer.is_running():
            viewer.sync()
