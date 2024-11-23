import torch
import numpy as np
from ultralytics.utils.ops import alloc_to_egoc_rot_matrix_torch


def get_object_keypoints(center_3d, size3d, roty):
    pts3d = get_box_corners(torch.tensor(size3d))
    return transform_to_camera(pts3d, torch.tensor(center_3d), torch.tensor(roty, device="cpu", dtype=torch.float32).unsqueeze(-1))


def get_3d_keypoints(center_3d, dep, size3d, alloc_rot_mat, calibs):
    calibs = calibs.unsqueeze(1).repeat(1, center_3d.shape[1], 1)
    locations = img_to_rect(center_3d, dep, calibs)
    boxes_object_frame = get_box_corners(size3d)
    #rotations = get_roty(center_3d, heading_bin, heading_res, calibs)
    egoc_rot_mat = alloc_to_egoc_rot_mat(alloc_rot_mat, center_3d, calibs)
    boxes_camera_frame = transform_to_camera(boxes_object_frame, locations, egoc_rot_mat)
    return boxes_camera_frame

def to_intrinsic_matrix(calibs):
    p2 = torch.zeros(calibs.shape[0], calibs.shape[1], 3, 4, dtype=torch.float32)
    p2[:, :, 0, 2] = calibs[:, :, 0]
    p2[:, :, 1, 2] = calibs[:, :, 1]
    p2[:, :, 0, 0] = calibs[:, :, 2]
    p2[:, :, 1, 1] = calibs[:, :, 3]
    p2[:, :, 0, 3] = calibs[:, :, 4] * -calibs[:, :, 2]
    p2[:, :, 1, 3] = calibs[:, :, 5] * -calibs[:, :, 3]
    p2[:, :, 2, 2] = 1
    return p2

def alloc_to_egoc_rot_mat(alloc_rot_mat, center_3d, calibs):
    return alloc_to_egoc_rot_matrix_torch(
        amodal_center=center_3d.flatten(end_dim=-2),
        alloc_rot_matrix=alloc_rot_mat.clone(),
        calib=to_intrinsic_matrix(calibs).to(center_3d.device)
    ).reshape(center_3d.shape[0], center_3d.shape[1], 3, 3)

def get_box_corners(size3d):
    hl, hw, hh = (size3d[..., 2].unsqueeze(-1) / 2, size3d[..., 1].unsqueeze(-1) / 2, size3d[..., 0].unsqueeze(-1) / 2)
    corners_x = torch.cat((hl, hl, -hl, -hl, hl, hl, -hl, -hl), dim=-1)
    corners_y = torch.cat((hw, -hw, -hw, hw, hw, -hw, -hw, hw), dim=-1)
    corners_z = torch.cat((-hh, -hh, -hh, -hh, hh, hh, hh, hh), dim=-1)
    box_corners = torch.cat((corners_x.unsqueeze(-1), corners_y.unsqueeze(-1), corners_z.unsqueeze(-1)), dim=-1)
    return box_corners


def get_omni_eval_corners(size3d):
    hl, hw, hh = (size3d[..., 2].unsqueeze(-1) / 2, size3d[..., 1].unsqueeze(-1) / 2, size3d[..., 0].unsqueeze(-1) / 2)
    return np.asarray([
        [-hl, -hw, hh],
        [hl, -hw, hh],
        [hl, -hw, -hh],
        [-hl, -hw, -hh],
        [-hl, hw, hh],
        [hl, hw, hh],
        [hl, hw, -hh],
        [-hl, hw, -hh]
    ])


def get_roty(center_3d, heading_bin, heading_res, calibs):
    if heading_bin.shape[-1] > 1:
        heading_bin = heading_bin.argmax(dim=-1)

    idx = torch.nn.functional.one_hot(heading_bin.long(), num_classes=12)

    if heading_res.shape[-1] > 1:
        heading_res = heading_res[idx.bool()].view(heading_bin.shape)
    alpha = class2angle(heading_bin, heading_res)
    ry = alpha2ry(alpha, center_3d[..., 0], calibs)
    return ry


def class2angle(cls, residual, num_heading_bin=12.0):
    angle_per_class = 2 * np.pi / num_heading_bin
    angle_center = cls * angle_per_class
    angle = angle_center + residual
    angle[angle > np.pi] = angle[angle > np.pi] - 2 * np.pi
    return angle


# From pytorch3d.transforms.rotation_conversions
def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


# From pytorch3d.transforms.rotation_conversion
def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def to_egoc_rot_mat(ry):
    rx = torch.full_like(ry, torch.pi / 2, device=ry.device)
    rz = torch.zeros_like(ry, device=ry.device)
    angles = torch.cat((rx, -ry, rz), dim=-1)
    return euler_angles_to_matrix(angles, convention="XYZ")


def alpha2ry(alpha, xs, calibs):
    cu, _, fu, _ = calibs.split((1, 1, 1, 3), dim=-1)
    if alpha.shape[-1] != 1:
        alpha = alpha.unsqueeze(-1)
    ry = alpha + torch.arctan2(xs.unsqueeze(-1) - cu, fu)
    ry[ry > torch.pi] = ry[ry > torch.pi] - 2 * torch.pi
    ry[ry < -torch.pi] = ry[ry < -torch.pi] + 2 * torch.pi
    return ry


def transform_to_camera(boxes_object_frame, locations, rot_mat):
    #rot_mat = to_egoc_rot_mat(ry)
    if len(rot_mat.shape) == 2:
        rot_mat = rot_mat.unsqueeze(0).unsqueeze(0).float()
        boxes_object_frame = boxes_object_frame.unsqueeze(0).unsqueeze(0).float()
    boxes = torch.einsum("bnij,bnjk->bnik", rot_mat.double(), boxes_object_frame.transpose(-2, -1).double()) + locations.unsqueeze(-1).double()
    return boxes.transpose(-2, -1)


def img_to_rect(center_3d, dep, calibs):
    cu, cv, fu, fv, tx, ty = calibs.split((1, 1, 1, 1, 1, 1), dim=-1)

    x = ((center_3d[..., 0].unsqueeze(-1) - cu) * dep) / fu + tx
    y = ((center_3d[..., 1].unsqueeze(-1) - cv) * dep) / fv + ty
    locations = torch.cat((x, y, dep), dim=-1)
    return locations