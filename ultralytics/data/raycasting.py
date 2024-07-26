import numpy as np
import torch
from typing import Tuple, Union


def project_to_image(pts_3d: np.ndarray, calibs: np.ndarray,
                     return_depth=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Projects the given points to the image.

    Args:
        pts_3d (np.ndarray): B x 3
        camera_matrix (np.ndarray): B x 3 x 4
        return_depth (bool, optional): Defaults to False.

    Returns:
        np.ndarray: B x 2
        np.ndarray (optional): B x P
    """
    pts_2d = np.asarray(pts_3d)
    calibs = np.asarray(calibs).reshape(-1, 3, 4)

    assert pts_3d.dtype in [np.float32, np.float64]
    assert calibs.dtype in [np.float32, np.float64]

    pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1))], axis=1)

    if len(calibs) == len(pts_3d):
        pts_2d = np.einsum('bij,bj->bi', calibs, pts_3d_homo)
    else:
        pts_2d = np.einsum('ij,bj->bi', calibs[0], pts_3d_homo)

    pts_2d, depth = pts_2d[:, :2] / pts_2d[:, 2:], pts_2d[:, 2]
    if return_depth:
        return pts_2d, depth
    else:
        return pts_2d


def project_to_image_torch(pts_3d: torch.Tensor, camera_matrix: torch.Tensor,
                           return_depth=False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Projects the given points to the image.

    Args:
        pts_3d (torch.Tensor): B x 3
        camera_matrix (torch.Tensor): B x 3 x 4
        return_depth (bool, optional): Defaults to False.

    Returns:
        torch.Tensor: B x 2
        torch.Tensor (optional): B x P
    """
    pts_3d = pts_3d.to(dtype=torch.float32)
    camera_matrix = camera_matrix.to(dtype=torch.float32)

    camera_matrix = camera_matrix.reshape(-1, 3, 4)
    pts_3d_homo = torch.cat([pts_3d, torch.ones_like(pts_3d[..., [0]])], axis=-1)

    if len(pts_3d_homo) == 0:
        pts_2d = torch.zeros((0, 2), device=pts_3d_homo.device)
        depth = torch.zeros(0, device=pts_3d_homo.device)
    else:
        with torch.cuda.amp.autocast(enabled=False):
            pts_2d = torch.einsum("bij,bj->bi", camera_matrix, pts_3d_homo)
        pts_2d, depth = pts_2d[:, :2] / pts_2d[:, 2:], pts_2d[:, 2]

    if return_depth:
        return pts_2d, depth
    else:
        return pts_2d


def project_multiple_points_torch(pts_3d: torch.Tensor, calib: torch.Tensor,
                                  return_depth=False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Projects P points per image with the same calibration matrix to the image.

    Args:
        pts_3d (torch.Tensor): B x P x 3
        calib (torch.Tensor): B x 3 x 4
        return_depth (bool, optional): Defaults to False.

    Returns:
        torch.Tensor: B x P x 2
        torch.Tensor (optional): B x P
    """
    assert pts_3d.ndim == 3
    n_points_per_img = pts_3d.shape[1]

    expanded_calib = calib[:, None].expand(-1, n_points_per_img, 3, 4).flatten(0, 1)
    pts_2d, depth = project_to_image_torch(pts_3d.flatten(0, 1), expanded_calib, return_depth=True)
    pts_2d = pts_2d.reshape(-1, n_points_per_img, 2)
    depth = depth.reshape(-1, n_points_per_img)

    if return_depth:
        return pts_2d, depth
    else:
        return pts_2d


def project_multiple_points(pts_3d: np.ndarray, calib: np.ndarray,
                            return_depth=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Projects P points per image with the same calibration matrix to the image.

    Args:
        pts_3d (np.ndarray): B x P x 3
        calib (np.ndarray): B x 3 x 4
        return_depth (bool, optional): Defaults to False.

    Returns:
        np.ndarray: B x P x 2
        np.ndarray (optional): B x P
    """
    assert pts_3d.ndim == 3
    n_points_per_img = pts_3d.shape[1]

    expanded_calib = np.repeat(calib[:, None], n_points_per_img, axis=1).reshape(-1, 3, 4)
    pts_2d, depth = project_to_image(pts_3d.reshape(-1, 3), expanded_calib, return_depth=True)
    pts_2d = pts_2d.reshape(-1, n_points_per_img, 2)
    depth = depth.reshape(-1, n_points_per_img)

    if return_depth:
        return pts_2d, depth
    else:
        return pts_2d


def unproject(pts_2d: np.ndarray, depths: Union[float, np.ndarray], calibs: np.ndarray):
    """Unprojects the given points from the image to 3D space.

    Args:
        pts_2d (np.ndarray): B x 2
        depths (np.ndarray): B
        calibs (np.ndarray): B x 3 x 4

    Returns:
        np.ndarray: B x 3
    """
    pts_2d = np.asarray(pts_2d)
    depths = np.asarray(depths)
    calibs = np.asarray(calibs)

    assert pts_2d.dtype in [np.float32, np.float64]
    assert depths.dtype in [np.float32, np.float64]
    assert calibs.dtype in [np.float32, np.float64]

    calibs = calibs.reshape(-1, 3, 4)

    z = depths - calibs[:, 2, 3]
    x = (pts_2d[:, 0] * depths - calibs[:, 0, 3] - calibs[:, 0, 2] * z) / calibs[:, 0, 0]
    y = (pts_2d[:, 1] * depths - calibs[:, 1, 3] - calibs[:, 1, 2] * z) / calibs[:, 1, 1]
    pts_3d = np.stack([x, y, z], axis=1)

    return pts_3d
