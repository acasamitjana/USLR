import itertools
import pdb
import copy

import numpy as np
import torch
from torch import nn

from utils import def_utils

#########################################
##   Linear Registration/Deformation   ##
#########################################

class InstanceRigidModel(nn.Module):

    def __init__(self, timepoints, reg_weight=0.001, cost='l1', device='cpu', torch_dtype=torch.float):
        super().__init__()

        self.device = device
        self.cost = cost
        self.reg_weight = reg_weight

        self.timepoints = timepoints
        self.N = len(timepoints)
        self.K = int(self.N * (self.N-1) / 2)

        # Parameters
        self.angle = torch.nn.Parameter(torch.zeros(3, self.N))
        self.translation = torch.nn.Parameter(torch.zeros(3, self.N))
        self.angle.requires_grad = True
        self.translation.requires_grad = True


    def _compute_matrix(self):

        angles = self.angle / 180 * np.pi

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        T = torch.zeros((4, 4, self.N))
        T[0, 0] = cos[2]*cos[1]
        T[1, 0] = sin[2]*cos[1]
        T[2, 0] = -sin[1]

        T[0, 1] = cos[2] * sin[1] * sin[0] - sin[2] * cos[0]
        T[1, 1] = sin[2] * sin[1] * sin[0] + cos[2] * cos[0]
        T[2, 1] = cos[1] * sin[0]

        T[0, 2] = cos[2] * sin[1] * cos[0] + sin[2] * sin[0]
        T[1, 2] = sin[2] * sin[1] * cos[0] - cos[2] * sin[0]
        T[2, 2] = cos[1] * cos[0]

        T[0, 3] = self.translation[0]# + self.tr0[0]
        T[1, 3] = self.translation[1]# + self.tr0[1]
        T[2, 3] = self.translation[2]# + self.tr0[2]
        T[3, 3] = 1

        #
        # for n in range(self.N):
        #
        #     T[..., n] = torch.chain_matmul(self.T0inv, T[..., n], self.T0)

        return T


    def _build_combinations(self, timepoints, latent_matrix):

        K = self.K
        if any([isinstance(t, str) for t in timepoints]):
            timepoints_dict = {
                t: it_t for it_t, t in enumerate(timepoints)
            }
        else:
            timepoints_dict = {
                t.id: it_t for it_t, t in enumerate(timepoints)
            }  # needed for non consecutive timepoints (if we'd like to skip one for whatever reason)


        Tij = torch.zeros((4, 4, K))

        k = 0
        for tp_ref, tp_flo in itertools.combinations(timepoints, 2):

            if not isinstance(tp_ref, str):
                t0 = timepoints_dict[tp_ref.id]
                t1 = timepoints_dict[tp_flo.id]
            else:
                t0 = timepoints_dict[tp_ref]
                t1 = timepoints_dict[tp_flo]


            T0k = latent_matrix[..., t0]
            T1k = latent_matrix[..., t1]

            Tij[..., k] = torch.matmul(T1k, torch.inverse(T0k))

            k += 1

        return Tij


    def _compute_log(self, Tij):

        K = Tij.shape[-1]
        R = Tij[:3, :3]
        Tr = Tij[:3, 3]

        logTij = torch.zeros((6, K))

        eps = 1e-6
        for k in range(K):
            t_norm = torch.arccos(torch.clamp((torch.trace(R[..., k]) - 1) / 2, min=-1+eps, max=1-eps)) + eps
            W = 1 / (2 * torch.sin(t_norm)) * (R[..., k] - R[..., k].T) * t_norm
            Vinv = torch.eye(3) - 0.5 * W + ((1 - (t_norm * torch.cos(t_norm / 2)) / (2 * torch.sin(t_norm / 2))) / t_norm ** 2) * W*W#torch.matmul(W, W)


            logTij[0, k] = 1 / (2 * torch.sin(t_norm)) * (R[..., k][2, 1] - R[..., k][1, 2]) * t_norm
            logTij[1, k] = 1 / (2 * torch.sin(t_norm)) * (R[..., k][0, 2] - R[..., k][2, 0]) * t_norm
            logTij[2, k] = 1 / (2 * torch.sin(t_norm)) * (R[..., k][1, 0] - R[..., k][0, 1]) * t_norm

            logTij[3:,k] = torch.matmul(Vinv, Tr[..., k])

        return logTij


    def forward(self, logRobs, timepoints):
        Ti = self._compute_matrix()
        Tij = self._build_combinations(timepoints, Ti)
        logTij = self._compute_log(Tij)
        logTi = self._compute_log(Ti)

        if self.cost == 'l1':
            loss = torch.sum(torch.sqrt(torch.sum((logTij - logRobs) ** 2, axis=0))) / self.K
        elif self.cost == 'l2':
            loss = torch.sum((logTij - logRobs) ** 2 + 1e-6) / self.K
        else:
            raise ValueError('Cost ' + self.cost + ' not valid. Choose \'l1\' of \'l2\'.' )
        loss += self.reg_weight * torch.sum(logTi**2) / self.K
        return loss


class InstanceRigidModelLOG(nn.Module):

    def __init__(self, timepoints, reg_weight=0.001, cost='l1', device='cpu', torch_dtype=torch.float):
        super().__init__()

        self.device = device
        self.cost = cost
        self.reg_weight = reg_weight

        self.timepoints = timepoints
        self.N = len(timepoints)
        self.K = int(self.N * (self.N-1) / 2)

        # Parameters
        self.angle = torch.nn.Parameter(torch.zeros(3, self.N))
        self.translation = torch.nn.Parameter(torch.zeros(3, self.N))
        self.angle.requires_grad = True
        self.translation.requires_grad = True


    def _compute_matrix(self):

        T = torch.zeros((4,4,self.N))
        for n in range(self.N):
            theta = torch.sqrt(torch.sum(self.angle[..., n]**2)) # torch.sum(torch.abs(self.angle))
            W = torch.zeros((3,3))
            W[1,0], W[0,1] = self.angle[2, n], -self.angle[2, n]
            W[0,2], W[2,0] = self.angle[1, n], -self.angle[1, n]
            W[2,1], W[1,2] = self.angle[0, n], -self.angle[0, n]
            V = torch.eye(3) + (1 - torch.cos(theta)) / (theta ** 2) * W + (theta - torch.sin(theta)) / (theta ** 3) * torch.matmul(W,W)

            T[:3, :3, n] = torch.eye(3) + torch.sin(theta) / theta * W      +      (1 - torch.cos(theta)) / (theta ** 2) * torch.matmul(W,W)
            T[:3, 3, n] = V @ self.translation[..., n]#torch.matmul(V, self.translation[..., n])
            T[3, 3, n] = 1

            #
            # for n in range(self.N):
            #
            #     T[..., n] = torch.chain_matmul(self.T0inv, T[..., n], self.T0)

        return T


    def _build_combinations(self, timepoints):

        K = self.K
        if any([isinstance(t, str) for t in timepoints]):
            timepoints_dict = {
                t: it_t for it_t, t in enumerate(timepoints)
            }
        else:
            timepoints_dict = {
                t.id: it_t for it_t, t in enumerate(timepoints)
            }  # needed for non consecutive timepoints (if we'd like to skip one for whatever reason)

        Tij = torch.zeros((6, K))

        k = 0
        for tp_ref, tp_flo in itertools.combinations(timepoints, 2):

            if not isinstance(tp_ref, str):
                t0 = timepoints_dict[tp_ref.id]
                t1 = timepoints_dict[tp_flo.id]
            else:
                t0 = timepoints_dict[tp_ref]
                t1 = timepoints_dict[tp_flo]
            Tij[:3, k] = self.angle[..., t1] - self.angle[..., t0]
            Tij[3:, k] = self.translation[..., t1] - self.translation[..., t0]

            k += 1

        return Tij


    def forward(self, logRobs, timepoints):

        logTij = self._build_combinations(timepoints)
        if self.cost == 'l1':
            loss = torch.sum(torch.sqrt(torch.sum((logTij - logRobs) ** 2, axis=0))) / self.K
        elif self.cost == 'l2':
            loss = torch.sum((logTij - logRobs) ** 2 + 1e-6) / self.K
        else:
            raise ValueError('Cost ' + self.cost + ' not valid. Choose \'l1\' of \'l2\'.' )
        loss += self.reg_weight * torch.sum(torch.sum(self.angle**2, axis=0) + torch.sum(self.translation**2, axis=0), axis=0) # / self.K

        return loss

class ST2Nonlinear(nn.Module):
    def __init__(self, obs_size, cp_size, factor=2, cost='l1', timepoints=None, init_T=None, reg_weight=1,
                 version=0, device='cpu'):

        super().__init__()
        self.obs_size = obs_size
        self.cost = cost
        self.device = device
        self.factor = factor
        self.reg_weight = reg_weight
        self.version = version

        if init_T is not None:
            self.N = len(init_T)
            self.T = torch.nn.ParameterDict({tid: torch.nn.Parameter(T) for tid, T in init_T.items()}).to(device)

        else:
            self.N = len(timepoints)
            self.T = torch.nn.ParameterDict({t.id: torch.nn.Parameter(torch.zeros((1, 3) + cp_size)) for t in timepoints}).to(device)

        # for T in self.T.values():
        self.T.requires_grad = True

        self.K = int(self.N * (self.N - 1) / 2)

        # ii = torch.arange(0, obs_size[0], dtype=torch.int, device=device)
        # jj = torch.arange(0, obs_size[1], dtype=torch.int, device=device)
        # kk = torch.arange(0, obs_size[2], dtype=torch.int, device=device)
        ii = torch.arange(0, cp_size[0], dtype=torch.int, device=device)
        jj = torch.arange(0, cp_size[1], dtype=torch.int, device=device)
        kk = torch.arange(0, cp_size[2], dtype=torch.int, device=device)
        self.grid = torch.unsqueeze(torch.stack(torch.meshgrid(ii, jj, kk, indexing='ij'), axis=0), 0)

        self.integrate = def_utils.VecInt(cp_size, int_steps=7).to(device)
        self.upscale = def_utils.RescaleTransform(cp_size, factor=self.factor).to(device) if self.factor != 1 else lambda x: x
        self.downscale = def_utils.RescaleTransform(cp_size, factor=1/self.factor).to(device) if self.factor != 1 else lambda x: x
        self.interp = def_utils.SpatialInterpolation().to(device)

    def _compose_fields(self, f1, f2):
        GG2 = self.grid + f1
        f2_int = self.interp(f2, GG2.clone())
        GG3 = GG2 + f2_int
        return GG3 -  self.grid

    def _build_combinations(self, tid_list):
        K = self.K
        k = 0
        R_hat = []#torch.zeros(self.obs_size + (3, K), device=self.device, requires_grad=True)
        for tp_id in tid_list:
            tp_ref, tp_flo = tp_id.split('_to_')
            # FIELD_REF = self.upscale(self.integrate(-self.T[tp_ref]))
            # FIELD_FLO = self.upscale(self.integrate(self.T[tp_flo]))
            FIELD_REF = self.integrate(-self.T[tp_ref])
            FIELD_FLO = self.integrate(self.T[tp_flo])
            R = self._compose_fields(FIELD_REF, FIELD_FLO)
            R_hat += [R]

            k += 1

        R_hat = torch.permute(torch.cat(R_hat, 0), (2, 3, 4, 1, 0))
        return R_hat

    def _get_difference(self, R, tid_list):
        R = torch.permute(R, (4, 3, 0, 1, 2))
        R_hat = []
        for it_tp, tp_id in enumerate(tid_list):
            tp_ref, tp_flo = tp_id.split('_to_')
            FIELD_FLO = self.upscale(self.integrate(-self.T[tp_flo])).float()
            FIELD_REF = self.upscale(self.integrate(self.T[tp_ref])).float()
            GG = self.grid + R[it_tp]
            f1_int = self.interp(FIELD_FLO, GG.clone())
            GG2 = GG + f1_int
            f2_int = self.interp(FIELD_REF, GG2.clone())
            GG3 = GG2 + f2_int
            R_hat += [GG3 -  self.grid]

        R_hat = torch.permute(torch.cat(R_hat, 0), (2, 3, 4, 1, 0))
        return R_hat

    def forward(self, R, tid_list, M=None):

        if self.version == 0:
            R_hat = self._build_combinations(tid_list)
            residue = R_hat - R

        elif self.version == 1:
            residue = self._get_difference(R, tid_list)

        else:
            R_hat = self._build_combinations(tid_list)
            residue = R_hat - R

        # pdb.set_trace()
        # import nibabel as nib
        # img = nib.Nifti1Image(R_hat[..., 10].cpu().detach().numpy(), np.eye(4))
        # nib.save(img, 'R_hat.prova.nii.gz')
        # img = nib.Nifti1Image(R[..., 10].cpu().detach().numpy(), np.eye(4))
        # nib.save(img, 'R.prova.nii.gz')

        if M is None:
            reduce_fn = torch.sum
        else:
            reduce_fn = lambda x: torch.sum(x*M)#/torch.sum(M)

        if self.cost == 'l1':
            loss = reduce_fn(torch.sqrt(torch.sum((residue) ** 2, axis=-2)))
        elif self.cost == 'l2':
            loss = reduce_fn(torch.sum((residue) ** 2, axis=-2))
        else:
            raise ValueError('Cost ' + self.cost + ' not valid. Choose \'l1\' of \'l2\'.' )

        # reg_loss = torch.sum(torch.sqrt(torch.sum(torch.sum(torch.cat([T for T in self.T.values()], dim=0), dim=0)**2, dim=0)))
        # loss += self.reg_weight * reg_loss

        # loss.backward()
        # for p in self.parameters():
        #     print(torch.max(torch.sqrt(torch.sum(p.grad ** 2, dim=1))))

        return loss
