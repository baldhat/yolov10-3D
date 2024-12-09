import torch
import numpy as np

class GradientBalancer(torch.nn.Module):

    def __init__(self, scaler, balancer="pcgrad", strategy="depVSrest", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = scaler
        self.strategy = strategy
        self.balancer = balancer
        if self.balancer == "pcgrad":
            self.balance = self.pc_grad
        elif self.balancer == "amtl":
            self.balance = self.amtl

        if self.strategy == "depVSrest":
            self.loss_groups = [
                [0, 1, 3, 4, 5, 6, 7, 9, 10, 11],
                [2, 8]
            ]
        elif self.strategy == "2dVS3d":
            self.loss_groups = [
                [0, 1, 6, 7],
                [2, 3, 4, 5, 8, 9, 10, 11]
            ]
        elif self.strategy == "distVSrest":
            self.loss_groups = [
                [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
                [6, 13]
            ]
        else:
            raise NotImplementedError("Unknown gradient balancing strategy")

    def step(self, model, loss_items):
        shared_params = list(model.model[:-1].parameters()) # backbone layers
        losses = self.aggregate_losses(loss_items)
        gradients = self.get_gradients_wrt_losses(shared_params, losses)
        self.set_shared_grad(shared_params, self.balance(gradients))


    def aggregate_losses(self, loss_items):
        assert sum(len(li) for li in  self.loss_groups) == loss_items.shape[0]

        losses = []
        for group in self.loss_groups:
            losses.append(torch.sum(loss_items[group]))
        return losses

    def get_gradients_wrt_losses(self, shared_params, losses):
        grads = []
        for i, cur_loss in enumerate(losses):
            for p in shared_params:
                if p.grad is not None:
                    p.grad.data.zero_()

            self.scaler.scale(cur_loss).backward(retain_graph=i < len(losses)-1)
            grad = torch.cat([p.grad.flatten().clone() if p.grad is not None else torch.zeros_like(p).flatten()
                              for p in shared_params])

            grads.append(grad)

        for p in shared_params:
            if p.grad is not None:
                p.grad.data.zero_()

        return torch.stack(grads, dim=0)

    @staticmethod
    def set_shared_grad(shared_params, grad_vec):
        offset = 0
        for p in shared_params:
            if p.grad is None:
                continue
            _offset = offset + p.grad.shape.numel()
            p.grad.data = grad_vec[offset:_offset].view_as(p.grad)
            offset = _offset

    @staticmethod
    def pc_grad(gradients):
        return RandomProjectionSolver.apply(gradients).sum(0)

    @staticmethod
    def amtl(gradients):
        grads, _, _ = ProcrustesSolver.apply(gradients.T.unsqueeze(0), "min")
        return grads[0].sum(-1)


########################### Taken from samsung mtl ############################

class RandomProjectionSolver:
    @staticmethod
    def apply(grads):
        assert (
            len(grads.shape) == 2
        ), f"Invalid shape of 'grads': {grads.shape}. Only 2D tensors are applicable"

        with torch.no_grad():
            order = torch.randperm(grads.shape[0])
            grads = grads[order]
            grads_task = grads

            def proj_grad(grad_task):
                for k in range(grads_task.shape[0]):
                    inner_product = torch.sum(grad_task * grads_task[k])
                    proj_direction = inner_product / (
                        torch.sum(grads_task[k] * grads_task[k]) + 1e-5
                    )
                    grad_task = (
                        grad_task
                        - torch.minimum(
                            proj_direction, torch.tensor(0.0).type_as(proj_direction)
                        )
                        * grads_task[k]
                    )
                return grad_task

            proj_grads = torch.stack(list(map(proj_grad, grads)), dim=0)

        return proj_grads


class ProcrustesSolver:
    @staticmethod
    def apply(grads, scale_mode='min'):
        assert (
            len(grads.shape) == 3
        ), f"Invalid shape of 'grads': {grads.shape}. Only 3D tensors are applicable"
        if not grads.isfinite().all():
            return grads, None, None

        with torch.no_grad():
            cov_grad_matrix_e = torch.matmul(grads.permute(0, 2, 1), grads)
            cov_grad_matrix_e = cov_grad_matrix_e.mean(0)

            singulars, basis = torch.linalg.eigh(cov_grad_matrix_e)
            tol = (
                torch.max(singulars)
                * max(cov_grad_matrix_e.shape[-2:])
                * torch.finfo().eps
            )
            rank = sum(singulars > tol)

            order = torch.argsort(singulars, dim=-1, descending=True)
            singulars, basis = singulars[order][:rank], basis[:, order][:, :rank]

            if scale_mode == 'min':
                weights = basis * torch.sqrt(singulars[-1]).view(1, -1)
            elif scale_mode == 'median':
                weights = basis * torch.sqrt(torch.median(singulars)).view(1, -1)
            elif scale_mode == 'rmse':
                weights = basis * torch.sqrt(singulars.mean())

            weights = weights / torch.sqrt(singulars).view(1, -1)
            weights = torch.matmul(weights, basis.T)
            grads = torch.matmul(grads, weights.unsqueeze(0))

            return grads, weights, singulars