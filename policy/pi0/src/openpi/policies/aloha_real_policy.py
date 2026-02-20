"""Aloha policy for real robot with single camera and single arm.

This is a modified version of aloha_policy.py specifically designed for:
- Single camera: cam_high only
- Single arm: Right arm only (7 DoF)
"""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_aloha_real_example() -> dict:
    """Creates a random input example for the Aloha real robot policy."""
    return {
        "state": np.ones((7, )),  # Only 7 DoF (right arm)
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class AlohaRealInputs(transforms.DataTransformFn):
    """Inputs for the Aloha real robot policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [7] (right arm only: 6 joints + 1 gripper)
    - actions: [action_horizon, 7]
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    # The expected cameras names. Only cam_high is expected.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "cam_high",
    )

    def __call__(self, data: dict) -> dict:
        data = _decode_aloha_real(data, adapt_to_pi=self.adapt_to_pi)

        # Get the state. We are padding from 7 to the model action dim.
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Only use head camera
        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Fill missing cameras with black images and False masks
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            # Always create black images for missing cameras
            images[dest] = np.zeros_like(base_image)
            image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv_real(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AlohaRealOutputs(transforms.DataTransformFn):
    """Outputs for the Aloha real robot policy."""

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        # Only return the first 7 dims (right arm only).
        actions = np.asarray(data["actions"][:, :7])
        return {"actions": _encode_actions_real(actions, adapt_to_pi=self.adapt_to_pi)}


def _joint_flip_mask_real() -> np.ndarray:
    """Used to convert between aloha and pi joint angles for right arm only."""
    # Only right arm: [j1, j2, j3, j4, j5, j6, gripper]
    # Pattern for right arm from original: [1, -1, -1, 1, 1, 1, 1]
    return np.array([1, -1, -1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)
    return _normalize(value, min_val=0.4, max_val=1.5)


def _gripper_from_angular(value):
    value = _unnormalize(value, min_val=0.4, max_val=1.5)
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return _normalize(value, min_val=0.4, max_val=1.5)


def _decode_aloha_real(data: dict, *, adapt_to_pi: bool = False) -> dict:
    # state is [right_arm_joint_angles(6), right_arm_gripper(1)]
    state = np.asarray(data["state"])
    state = _decode_state_real(state, adapt_to_pi=adapt_to_pi)

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data


def _decode_state_real(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        state = _joint_flip_mask_real() * state
        # Reverse the gripper transformation (last element is gripper).
        state[6] = _gripper_to_angular(state[6])
    return state


def _encode_actions_real(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        actions = _joint_flip_mask_real() * actions
        # Transform gripper (last column is gripper).
        actions[:, 6] = _gripper_from_angular(actions[:, 6])
    return actions


def _encode_actions_inv_real(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask_real() * actions
        actions[:, 6] = _gripper_from_angular_inv(actions[:, 6])
    return actions
