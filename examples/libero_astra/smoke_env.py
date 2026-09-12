"""Check the real Panda environment and save images without making model calls."""

import json
import time
from pathlib import Path

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image
from run import grid_image
from scipy.spatial.transform import Rotation

root = Path(__file__).resolve().parents[2] / "artifacts/astra_libero/smoke"
root.mkdir(parents=True, exist_ok=True)
suite = benchmark.get_benchmark_dict()["libero_goal"]()
task = suite.get_task(5)
print("task:", task.language, flush=True)
start = time.perf_counter()
env = OffScreenRenderEnv(
    bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
    camera_heights=384,
    camera_widths=384,
    camera_names=["agentview", "robot0_eye_in_hand", "birdview"],
)
env.seed(7)
env.reset()
camera_id = env.sim.model.camera_name2id("birdview")
env.sim.model.cam_pos[camera_id] = [0.0, 0.0, 2.1]
env.sim.model.cam_quat[camera_id] = [1.0, 0.0, 0.0, 0.0]
obs = env.set_init_state(suite.get_task_init_states(5)[0])
for _ in range(10):
    obs, reward, done, info = env.step([0.0] * 6 + [-1.0])
for name in ["agentview", "robot0_eye_in_hand", "birdview"]:
    Image.fromarray(np.ascontiguousarray(obs[name + "_image"][::-1])).save(root / f"{name}.png")
controller = env.robots[0].controller
grid_image(env, obs).save(root / "grid.png")
initial_position = obs["robot0_eef_pos"].copy()
target = initial_position + np.array([0.06, 0.02, 0.02])
target_rotation = Rotation.from_quat(obs["robot0_eef_quat"]).as_matrix()
motor_actions = []
for _ in range(40):
    rotation = Rotation.from_quat(obs["robot0_eef_quat"]).as_matrix()
    error = Rotation.from_matrix(target_rotation @ rotation.T).as_rotvec()
    action = np.r_[
        np.clip((target - obs["robot0_eef_pos"]) / 0.05, -0.5, 0.5),
        np.clip(error / 0.5, -0.5, 0.5),
        -1.0,
    ]
    obs, _, _, _ = env.step(action.tolist())
    motor_actions.append(action.tolist())
error_m = float(np.linalg.norm(target - obs["robot0_eef_pos"]))
assert error_m < 0.005, f"Cartesian controller failed: error {error_m} m"
Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1])).save(root / "after_motion.png")
data = {
    "elapsed_s": time.perf_counter() - start,
    "action_spec": [np.asarray(x).tolist() for x in env.env.action_spec],
    "robot_state": {
        k: np.asarray(v).tolist()
        for k, v in obs.items()
        if k.startswith("robot0_") and "image" not in k
    },
    "output_min": controller.output_min.tolist(),
    "output_max": controller.output_max.tolist(),
    "eef_site_id": env.robots[0].eef_site_id,
    "table_offset": list(env.env.table_offset),
    "camera_names": list(env.sim.model.camera_names),
    "control_smoke": {
        "initial_position": initial_position.tolist(),
        "target": target.tolist(),
        "actual": obs["robot0_eef_pos"].tolist(),
        "error_m": error_m,
        "actions": motor_actions,
    },
}
(root / "state.json").write_text(json.dumps(data, indent=2))
print(json.dumps(data, indent=2), flush=True)
env.close()
