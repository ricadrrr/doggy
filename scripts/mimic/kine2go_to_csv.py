"""Convert a Kine2Go reference trajectory to the CSV format used by csv_to_npz.py.

The output has 19 columns per frame:

    base position xyz (3), base quaternion xyzw (4), Go2 joint positions (12)

Example:

    python scripts/mimic/kine2go_to_csv.py \
        datasets/kine2go/ai4_dog_canter/raw/motion.npy \
        datasets/kine2go/ai4_dog_canter/ai4_dog_canter_60hz.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


KINE2GO_FRAME_SIZE = 61
GO2_JOINT_COUNT = 12

# Kine2Go stores the 12 actuated joints in Genesis local-DOF order:
#   FL/FR/RL/RR hips, then thighs, then calves.
# unitree_rl_lab's CSV uses Unitree SDK order:
#   FR leg, FL leg, RR leg, RL leg; hip/thigh/calf within each leg.
GENESIS_TO_SDK_JOINT_INDICES = np.array([1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10])


def convert_motion(
    input_file: Path,
    output_file: Path,
    frame_start: int | None = None,
    frame_end: int | None = None,
    repeat_cycles: int = 1,
) -> tuple[int, int]:
    """Convert one Kine2Go ``motion.npy`` file and return its output shape."""
    motion = np.load(input_file)

    if motion.ndim != 2 or motion.shape[1] != KINE2GO_FRAME_SIZE:
        raise ValueError(
            f"Expected Kine2Go motion with shape (T, {KINE2GO_FRAME_SIZE}), got {motion.shape}."
        )
    if not np.isfinite(motion).all():
        raise ValueError("Input motion contains NaN or infinite values.")

    start = 0 if frame_start is None else frame_start
    end = motion.shape[0] if frame_end is None else frame_end
    if not 0 <= start < end <= motion.shape[0]:
        raise ValueError(f"Invalid frame range [{start}:{end}] for {motion.shape[0]} input frames.")
    if repeat_cycles < 1:
        raise ValueError(f"repeat_cycles must be at least 1, got {repeat_cycles}.")
    if repeat_cycles > 1 and end >= motion.shape[0]:
        raise ValueError("Repeating a cycle requires one source frame after frame_end.")

    cycle_motion = motion[start:end]

    # Kine2Go schema:
    #   [6:18]  -- Go2 joints in Genesis local-DOF order
    #   [48:51] -- base position xyz
    #   [51:55] -- base quaternion wxyz
    joint_pos_genesis = cycle_motion[:, 6:18]
    joint_pos = joint_pos_genesis[:, GENESIS_TO_SDK_JOINT_INDICES]
    base_pos = cycle_motion[:, 48:51]
    base_quat_wxyz = cycle_motion[:, 51:55]

    if joint_pos.shape[1] != GO2_JOINT_COUNT:
        raise ValueError(f"Expected {GO2_JOINT_COUNT} Go2 joints, got {joint_pos.shape[1]}.")

    quat_norm = np.linalg.norm(base_quat_wxyz, axis=1)
    if not np.allclose(quat_norm, 1.0, atol=1e-4):
        max_error = float(np.max(np.abs(quat_norm - 1.0)))
        raise ValueError(f"Base quaternions are not normalized (max norm error: {max_error:.3e}).")

    if repeat_cycles > 1:
        endpoint_pos = motion[end, 48:51]
        endpoint_quat_wxyz = motion[end, 51:55]
        base_pos, base_quat_wxyz = repeat_planar_cycle(
            base_pos,
            base_quat_wxyz,
            endpoint_pos,
            endpoint_quat_wxyz,
            repeat_cycles,
        )
        joint_pos = np.tile(joint_pos, (repeat_cycles, 1))

    # csv_to_npz.py expects xyzw and converts it back to Isaac Lab's wxyz.
    base_quat_xyzw = base_quat_wxyz[:, [1, 2, 3, 0]]
    output = np.concatenate((base_pos, base_quat_xyzw, joint_pos), axis=1)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_file, output, delimiter=",", fmt="%.10g")
    return output.shape


def yaw_from_wxyz(quaternion: np.ndarray) -> float:
    """Return the yaw angle of one wxyz quaternion."""
    w, x, y, z = quaternion
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def yaw_quaternion_wxyz(yaw: float) -> np.ndarray:
    """Return a wxyz quaternion containing only a yaw rotation."""
    half_yaw = 0.5 * yaw
    return np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)])


def quaternion_multiply_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply two wxyz quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def repeat_planar_cycle(
    cycle_pos: np.ndarray,
    cycle_quat_wxyz: np.ndarray,
    endpoint_pos: np.ndarray,
    endpoint_quat_wxyz: np.ndarray,
    repeat_cycles: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Repeat a cycle while accumulating its horizontal translation and yaw."""
    start_pos = cycle_pos[0]
    start_yaw = yaw_from_wxyz(cycle_quat_wxyz[0])
    endpoint_yaw = yaw_from_wxyz(endpoint_quat_wxyz)
    yaw_step = float(np.arctan2(np.sin(endpoint_yaw - start_yaw), np.cos(endpoint_yaw - start_yaw)))

    rotation_step = np.array(
        [
            [np.cos(yaw_step), -np.sin(yaw_step)],
            [np.sin(yaw_step), np.cos(yaw_step)],
        ]
    )
    # This left-side transform maps the original cycle start exactly onto its endpoint.
    translation_step = endpoint_pos[:2] - rotation_step @ start_pos[:2]

    repeated_pos = []
    repeated_quat = []
    rotation_acc = np.eye(2)
    translation_acc = np.zeros(2)
    yaw_acc = 0.0

    for _ in range(repeat_cycles):
        transformed_pos = cycle_pos.copy()
        transformed_pos[:, :2] = (rotation_acc @ cycle_pos[:, :2].T).T + translation_acc
        repeated_pos.append(transformed_pos)

        yaw_rotation = yaw_quaternion_wxyz(yaw_acc)
        transformed_quat = np.stack(
            [quaternion_multiply_wxyz(yaw_rotation, quat) for quat in cycle_quat_wxyz]
        )
        repeated_quat.append(transformed_quat)

        translation_acc = rotation_step @ translation_acc + translation_step
        rotation_acc = rotation_step @ rotation_acc
        yaw_acc += yaw_step

    positions = np.concatenate(repeated_pos, axis=0)
    quaternions = np.concatenate(repeated_quat, axis=0)

    # q and -q describe the same rotation, but keeping neighboring signs aligned
    # prevents interpolation from taking an unnecessary long path.
    for index in range(1, quaternions.shape[0]):
        if np.dot(quaternions[index - 1], quaternions[index]) < 0.0:
            quaternions[index] *= -1.0

    return positions, quaternions


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Kine2Go motion.npy to unitree_rl_lab CSV.")
    parser.add_argument("input_file", type=Path, help="Path to Kine2Go motion.npy.")
    parser.add_argument("output_file", type=Path, help="Path to the output CSV.")
    parser.add_argument("--frame-start", type=int, help="First input frame to include (zero-based, inclusive).")
    parser.add_argument("--frame-end", type=int, help="Last input frame boundary (zero-based, exclusive).")
    parser.add_argument(
        "--repeat-cycles",
        type=int,
        default=1,
        help="Number of cycles to generate while accumulating horizontal translation and yaw.",
    )
    args = parser.parse_args()

    shape = convert_motion(
        args.input_file,
        args.output_file,
        args.frame_start,
        args.frame_end,
        args.repeat_cycles,
    )
    print(f"Saved {shape[0]} frames x {shape[1]} columns to: {args.output_file}")


if __name__ == "__main__":
    main()
