import robosuite as suite
from robosuite.wrappers import GymWrapper
import xml.etree.ElementTree as ET

if __name__ == "__main__":

    tree = ET.parse("/home/guojun/winshare/evolution/robosuite/robosuite/models/assets/grippers/robotiq_gripper_85.xml")  # 解析 XML 文件
    root = tree.getroot()

    for geom in root.findall(".//geom[@name='{}']".format("left_fingertip_visual")):
        size = [float(e) for e in geom.get("size").split(" ")]
        pos = [float(e) for e in geom.get("pos").split(" ")]
    for geom in root.findall(".//geom[@name='{}']".format("left_fingerpad_collision")):
        size2 = [float(e) for e in geom.get("size").split(" ")]
        pos2 = [float(e) for e in geom.get("pos").split(" ")]

    sx, sy, sz = 1.5, 2.5, 2.3
    new_size = [size[0] * sx, size[1] * sy, size[2] * sz]
    new_pos = [pos[0], pos[1] - new_size[1] + size[1], pos[2] + new_size[2] - size[2]]
    new_size2 = [size2[0] * sx, size[1] , size2[2] * sz]
    new_pos2 = [pos2[0], pos2[1] - (new_size[1] - size[1]) * 2, pos2[2] + new_size[2] - size[2]]
    for geom in root.iter("geom"):
        if geom.get("name") in ["left_fingertip_visual", "left_fingertip_collision", "right_fingertip_visual", "right_fingertip_collision"]:
            geom.set("size","{:.5f} {:.5f} {:.5f}".format(*new_size))
            geom.set("pos","{:.5f} {:.5f} {:.5f}".format(*new_pos))

        if geom.get("name") in ["right_fingerpad_visual", "right_fingerpad_collision", "left_fingerpad_visual", "left_fingerpad_collision"]:
            geom.set("size","{:.5f} {:.5f} {:.5f}".format(*new_size2))
            geom.set("pos","{:.5f} {:.5f} {:.5f}".format(*new_pos2))

    tree.write("/home/guojun/winshare/evolution/robosuite/robosuite/models/assets/grippers/evo.xml", encoding='utf-8', xml_declaration=True)

    # Notice how the environment is wrapped by the wrapper
    env = GymWrapper(
        suite.make(
            "Lift",
            robots="UR5e",  # use Sawyer robot
            gripper_types="EvoGripper",
            use_camera_obs=False,  # do not use pixel observations
            has_offscreen_renderer=False,  # not needed since not using pixel obs
            has_renderer=True,  # make sure we can render to the screen
            reward_shaping=True,  # use dense rewards
            control_freq=20,  # control should happen fast enough so that simulation looks smooth
            ignore_done=True,
            render_camera=None
        )
    )

    env.reset(seed=0)

    while True:
        env.render()
        action = env.action_space.sample() * 0
        observation, reward, terminated, truncated, info = env.step(action)