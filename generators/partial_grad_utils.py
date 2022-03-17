import torch

from einops import rearrange


def scatter_points(idx_grad, points_grad, idx_no_grad, points_no_grad, num_points):

    points_all = torch.zeros(
        points_grad.shape[0],
        num_points,
        points_grad.shape[-1],
        device=points_grad.device,
        dtype=points_grad.dtype,
    )

    idx_grad = rearrange(idx_grad, "n -> 1 n 1")
    idx_grad_out = idx_grad.expand(points_grad.shape[0], -1, points_grad.shape[-1])
    points_all.scatter_(dim=1, index=idx_grad_out, src=points_grad)

    idx_no_grad = rearrange(idx_no_grad, "n -> 1 n 1")
    idx_no_grad_out = idx_no_grad.expand(
        points_no_grad.shape[0], -1, points_no_grad.shape[-1]
    )
    points_all.scatter_(dim=1, index=idx_no_grad_out, src=points_no_grad)
    return points_all


def gather_points(points, idx_grad):
    """

    :param points: (b, n, c) or (b, n, s, c)
    :param idx_grad:
    :return:
    """
    if points.dim() == 4:
        idx_grad = rearrange(idx_grad, "n -> 1 n 1 1")
        idx_grad = idx_grad.expand(
            points.shape[0], -1, points.shape[-2], points.shape[-1]
        )
        sampled_points = torch.gather(points, dim=1, index=idx_grad, sparse_grad=True)
    elif points.dim() == 3:
        idx_grad = rearrange(idx_grad, "n -> 1 n 1")
        idx_grad = idx_grad.expand(points.shape[0], -1, points.shape[-1])
        sampled_points = torch.gather(points, dim=1, index=idx_grad, sparse_grad=True)
    else:
        assert 0
    return sampled_points
