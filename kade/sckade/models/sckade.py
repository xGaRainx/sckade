import os
import copy
import itertools
import collections
from math import ceil
from typing import Any, List, Mapping, NoReturn, Optional, Tuple, Union

import ignite
import numpy as np
import pandas as pd
import torch
import torch.distributions as D
import torch.nn.functional as F
from torch import Module, linalg as LA
from anndata import AnnData

from ..utils import *
from ..num import EPS
from .base import Trainer, Model, TrainingPlugin
from .data import AnnDataset, ArrayDataset, DataLoader
from .kan import KANLinear
from .plugins import EarlyStopping, LRScheduler, Tensorboard
from .prob import ZINB


#----------------------- Component interface definitions -----------------------
class Encoder(torch.nn.Module):

    def __init__(
            self, in_features: int, out_features: int,
            h_depth: int = 2, h_dim: int = 256,
            dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.h_depth = h_depth
        ptr_dim = in_features
        for layer in range(self.h_depth):
            setattr(self, f"KANlinear_{layer}", KANLinear(ptr_dim, h_dim))
            setattr(self, f"bn_{layer}", torch.nn.BatchNorm1d(h_dim))

            ptr_dim = h_dim
        self.loc = torch.nn.Linear(ptr_dim, out_features)
        self.std_lin = torch.nn.Linear(ptr_dim, out_features)


    TOTAL_COUNT = 1e4

    def compute_l(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1, keepdim=True)

    def normalize(
            self, x: torch.Tensor, l: torch.Tensor
    ) -> torch.Tensor:
        return (x * (self.TOTAL_COUNT / l)).log1p()
    
    def forward(  # pylint: disable=arguments-differ
            self, x: torch.Tensor, xrep: torch.Tensor,
            lazy_normalizer: bool = True
    ) -> Tuple[D.Normal, Optional[torch.Tensor]]:

        if xrep.numel():
            l = None if lazy_normalizer else self.compute_l(x)
            ptr = xrep
        else:
            l = self.compute_l(x)
            ptr = self.normalize(x, l)
        for layer in range(self.h_depth):
            ptr = getattr(self, f"KANlinear_{layer}")(ptr)
            ptr = getattr(self, f"bn_{layer}")(ptr)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return D.Normal(loc, std), l
    

class Decoder(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int, n_batches: int = 1) -> None:  # pylint: disable=unused-argument
        super().__init__()
        self.v = torch.nn.Parameter(torch.zeros(out_features, in_features))
        self.scale_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.log_theta = torch.nn.Parameter(torch.zeros(n_batches, out_features))

    def forward(
            self, u: torch.Tensor,
            b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> D.NegativeBinomial:
        scale = F.softplus(self.scale_lin[b])
        logit_mu = scale * (u @ self.v.t()) + self.bias[b]
        mu = F.softmax(logit_mu, dim=1) * l
        log_theta = self.log_theta[b]
        return D.NegativeBinomial(
            log_theta.exp(),
            logits=(mu + EPS).log() - log_theta
        )
    

class Discriminator(torch.nn.Sequential):

    def __init__(
            self, in_features: int, out_features: int, n_batches: int = 0,
            h_depth: int = 2, h_dim: Optional[int] = 256,
            dropout: float = 0.2
    ) -> None:
        self.n_batches = n_batches
        od = collections.OrderedDict()
        ptr_dim = in_features + self.n_batches
        for layer in range(h_depth):
            od[f"linear_{layer}"] = torch.nn.Linear(ptr_dim, h_dim)
            od[f"act_{layer}"] = torch.nn.LeakyReLU(negative_slope=0.2)
            od[f"dropout_{layer}"] = torch.nn.Dropout(p=dropout)
            ptr_dim = h_dim
        od["pred"] = torch.nn.Linear(ptr_dim, out_features)
        super().__init__(od)

    def forward(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        if self.n_batches:
            b_one_hot = F.one_hot(b, num_classes=self.n_batches)
            x = torch.cat([x, b_one_hot], dim=1)
        return super().forward(x)
    
    
class Prior(torch.nn.Module):
    
    def __init__(
            self, loc: float = 0.0, std: float = 1.0
    ) -> None:
        super().__init__()
        loc = torch.as_tensor(loc, dtype=torch.get_default_dtype())
        std = torch.as_tensor(std, dtype=torch.get_default_dtype())
        self.register_buffer("loc", loc)
        self.register_buffer("std", std)

    def forward(self) -> D.Normal:
        return D.Normal(self.loc, self.std)


class Classifier(torch.nn.Linear):

    """
    Linear label classifier

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    """


#------------------------ Network interface definition -------------------------

class SCKADE(torch.nn.Module):
    def __init__(
            self,
            x2u: Mapping[str, Encoder],
            u2x: Mapping[str, Decoder],
            du: Discriminator, prior: Prior,
            u2c: Optional[Classifier] = None
            
    ) -> None:
        super().__init__()
        self.u2c = u2c.to(self.device) if u2c else None

        if not set(x2u.keys()) == set(u2x.keys()) != set():
            raise ValueError(
                "`x2u`, `u2x` should share the same keys "
                "and non-empty!"
            )
        self.keys = list(x2u.keys())  # Keeps a specific order

        self.x2u = torch.nn.ModuleDict(x2u)
        self.u2x = torch.nn.ModuleDict(u2x)
        self.du = du
        self.prior = prior

        self.device = autodevice()
        self.u2c = u2c.to(self.device) if u2c else None

    @property
    def device(self) -> torch.device:

        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        self._device = device
        self.to(self._device)

    def forward(self) -> NoReturn:
        """
        Invalidated forward operation
        """
        raise RuntimeError("This module does not support forward operation!")
    

#----------------------------- Trainer definition ------------------------------

DataTensors = Tuple[
    Mapping[str, torch.Tensor],  # x (data)
    Mapping[str, torch.Tensor],  # xrep (alternative input data)
    Mapping[str, torch.Tensor],  # xbch (data batch)
    Mapping[str, torch.Tensor],  # xlbl (data label)
    Mapping[str, torch.Tensor],  # xdwt (modality discriminator sample weight)
    Mapping[str, torch.Tensor],  # xflag (modality indicator)
]  # Specifies the data format of input to SCGLUETrainer.compute_losses


@logged
class SCKADETrainer(Trainer):

    """
    Parameters
    ----------
    net
        :class:`SCKADE` network to be trained
    lam_data
        Data weight
    lam_kl
        KL weight
    lam_align
        Adversarial alignment weight
    lam_sup
        Cell type supervision weight
    normalize_u
        Whether to L2 normalize cell embeddings before decoder
    modality_weight
        Relative modality weight (indexed by modality name)
    optim
        Optimizer
    lr
        Learning rate
    **kwargs
        Additional keyword arguments are passed to the optimizer constructor
    """

    BURNIN_NOISE_EXAG: float = 1.5  # Burn-in noise exaggeration

    def __init__(
            self, net: SCKADE, lam_data: float = None,
            lam_kl: float = None, lam_align: float = None,
            modality_weight: Mapping[str, float] = None,
            optim: str = None, lr: float = None,

            lam_sup: float = None, normalize_u: bool = None, **kwargs
    ) -> None:
        required_kwargs = (
            "lam_data", "lam_kl", "lam_align",
            "modality_weight", "optim", "lr",
            "lam_sup", "normalize_u"
        )
        for required_kwarg in required_kwargs:
            if locals()[required_kwarg] is None:
                raise ValueError(f"`{required_kwarg}` must be specified!")
        super().__init__(net)

        for k in self.net.keys:
            self.required_losses += [f"x_{k}_nll", f"x_{k}_kl", f"x_{k}_elbo"]
        self.required_losses += ["dsc_loss", "vae_loss", "gen_loss"]
        self.earlystop_loss = "vae_loss"

        self.lam_data = lam_data
        self.lam_kl = lam_kl
        self.lam_align = lam_align
        if min(modality_weight.values()) < 0:
            raise ValueError("Modality weight must be non-negative!")
        normalizer = sum(modality_weight.values()) / len(modality_weight)
        self.modality_weight = {k: v / normalizer for k, v in modality_weight.items()}

        self.lr = lr
        
        if net.u2c:
            self.required_losses.append("sup_loss")
            self.vae_optim = getattr(torch.optim, optim)(
                itertools.chain(
                    self.net.x2u.parameters(),
                    self.net.u2x.parameters(),
                    self.net.u2c.parameters(),
                ), lr=self.lr, **kwargs
            )
        else:
            self.vae_optim = getattr(torch.optim, optim)(
                itertools.chain(
                    self.net.x2u.parameters(),
                    self.net.u2x.parameters(),
                ), lr=self.lr, **kwargs
            )
        self.dsc_optim = getattr(torch.optim, optim)(
            self.net.du.parameters(), lr=self.lr, **kwargs
        )

        self.align_burnin: Optional[int] = None
        self.lam_sup = lam_sup
        self.normalize_u = normalize_u
        self.freeze_u = False

    @property
    def freeze_u(self) -> bool:
        """
        Whether to freeze cell embeddings
        """
        return self._freeze_u

    @freeze_u.setter
    def freeze_u(self, freeze_u: bool) -> None:
        self._freeze_u = freeze_u
        for item in itertools.chain(self.net.x2u.parameters(), self.net.du.parameters()):
            item.requires_grad_(not self._freeze_u)

    def format_data(self, data: List[torch.Tensor]) -> DataTensors:
        """
        Format data tensors

        Note
        ----
        The data dataset should contain data arrays for each modality,
        followed by alternative input arrays for each modality,
        in the same order as modality keys of the network.
        """
        device = self.net.device
        keys = self.net.keys
        K = len(keys)
        x, xrep, xbch, xlbl, xdwt = \
            data[0:K], data[K:2*K], data[2*K:3*K], data[3*K:4*K], data[4*K:5*K]
        x = {
            k: x[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xrep = {
            k: xrep[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xbch = {
            k: xbch[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xlbl = {
            k: xlbl[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xdwt = {
            k: xdwt[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xflag = {
            k: torch.as_tensor(
                i, dtype=torch.int64, device=device
            ).expand(x[k].shape[0])
            for i, k in enumerate(keys)
        }
        return x, xrep, xbch, xlbl, xdwt, xflag
    
    def compute_losses(
            self, data: DataTensors, epoch: int, dsc_only: bool = False
    ) -> Mapping[str, torch.Tensor]:  # pragma: no cover
        """
        Compute loss functions

        Parameters
        ----------
        data
            Data tensors
        epoch
            Current epoch number
        dsc_only
            Whether to compute the discriminator loss only

        Returns
        -------
        loss_dict
            Dict containing loss values
        """
        net = self.net
        x, xrep, xbch, xlbl, xdwt, xflag = data

        u, l = {}, {}
        for k in net.keys:
            u[k], l[k] = net.x2u[k](x[k], xrep[k], lazy_normalizer=dsc_only)

        usamp = {k: u[k].rsample() for k in net.keys}
        
        if self.normalize_u:
            usamp = {k: F.normalize(usamp[k], dim=1) for k in net.keys}
        prior = net.prior()
        
        u_cat = torch.cat([u[k].mean for k in net.keys])
        xbch_cat = torch.cat([xbch[k] for k in net.keys])
        xdwt_cat = torch.cat([xdwt[k] for k in net.keys])
        xflag_cat = torch.cat([xflag[k] for k in net.keys])
        anneal = max(1 - (epoch - 1) / self.align_burnin, 0) \
            if self.align_burnin else 0
        if anneal:
            noise = D.Normal(0, u_cat.std(axis=0)).sample((u_cat.shape[0], ))
            u_cat = u_cat + (anneal * self.BURNIN_NOISE_EXAG) * noise
        
        dsc_loss = F.cross_entropy(net.du(u_cat, xbch_cat), xflag_cat, reduction="none")
        dsc_loss = (dsc_loss * xdwt_cat).sum() / xdwt_cat.numel()

        if dsc_only:
            return {"dsc_loss": self.lam_align * dsc_loss}
        
        if net.u2c:
            xlbl_cat = torch.cat([xlbl[k] for k in net.keys])
            lmsk = xlbl_cat >= 0
            sup_loss = F.cross_entropy(
                net.u2c(u_cat[lmsk]), xlbl_cat[lmsk], reduction="none"
            ).sum() / max(lmsk.sum(), 1)
        else:
            sup_loss = torch.tensor(0.0, device=self.net.device)

        x_nll = {
            k: -net.u2x[k](
                usamp[k], xbch[k], l[k]
            ).log_prob(x[k]).mean()
            for k in net.keys
        }

        x_kl = {
            k: D.kl_divergence(
                u[k], prior
            ).sum(dim=1).mean() / x[k].shape[1]
            for k in net.keys
        }

        x_elbo = {
            k: x_nll[k] \
                + self.lam_kl * x_kl[k]
            for k in net.keys
        }

        x_elbo_sum = sum(self.modality_weight[k] * x_elbo[k] for k in net.keys)

        vae_loss = self.lam_data * x_elbo_sum \
            + self.lam_sup * sup_loss
        
        gen_loss = vae_loss - self.lam_align * dsc_loss

        losses = {
            "dsc_loss": dsc_loss, "vae_loss": vae_loss, "gen_loss": gen_loss,
        }

        for k in net.keys:
            losses.update({
                f"x_{k}_nll": x_nll[k],
                f"x_{k}_kl": x_kl[k],
                f"x_{k}_elbo": x_elbo[k]
            })

        if net.u2c:
            losses["sup_loss"] = sup_loss
        return losses
    
    def train_step(
            self, engine: ignite.engine.Engine, data: List[torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        self.net.train()
        data = self.format_data(data)
        epoch = engine.state.epoch

        if self.freeze_u:
            self.net.x2u.apply(freeze_running_stats)
            self.net.du.apply(freeze_running_stats)
        else:  # Discriminator step
            losses = self.compute_losses(data, epoch, dsc_only=True)
            self.net.zero_grad(set_to_none=True)
            losses["dsc_loss"].backward()  # Already scaled by lam_align
            self.dsc_optim.step()

        # Generator step
        losses = self.compute_losses(data, epoch)
        self.net.zero_grad(set_to_none=True)
        losses["gen_loss"].backward()
        self.vae_optim.step()

        return losses
    
    @torch.no_grad()
    def val_step(
            self, engine: ignite.engine.Engine, data: List[torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        self.net.eval()
        data = self.format_data(data)
        return self.compute_losses(data, engine.state.epoch)
    
    def fit(  # pylint: disable=arguments-renamed
            self, data: ArrayDataset, 
            val_split: float = None, data_batch_size: int = None,
            align_burnin: int = None, safe_burnin: bool = True,
            max_epochs: int = None, patience: Optional[int] = None,
            reduce_lr_patience: Optional[int] = None,
            wait_n_lrs: Optional[int] = None,
            random_seed: int = None, directory: Optional[os.PathLike] = None,
            plugins: Optional[List[TrainingPlugin]] = None
    ) -> None:
        """
        Fit network

        Parameters
        ----------
        data
            Data dataset
        val_split
            Validation split
        data_batch_size
            Number of samples in each data minibatch
        align_burnin
            Number of epochs to wait before starting alignment
        safe_burnin
            Whether to postpone learning rate scheduling and earlystopping
            until after the burnin stage
        max_epochs
            Maximal number of epochs
        patience
            Patience of early stopping
        reduce_lr_patience
            Patience to reduce learning rate
        wait_n_lrs
            Wait n learning rate scheduling events before starting early stopping
        random_seed
            Random seed
        directory
            Directory to store checkpoints and tensorboard logs
        plugins
            Optional list of training plugins
        """
        required_kwargs = (
            "val_split", "data_batch_size",
            "align_burnin", "max_epochs", "random_seed"
        )
        for required_kwarg in required_kwargs:
            if locals()[required_kwarg] is None:
                raise ValueError(f"`{required_kwarg}` must be specified!")
        if patience and reduce_lr_patience and reduce_lr_patience >= patience:
            self.logger.warning(
                "Parameter `reduce_lr_patience` should be smaller than `patience`, "
                "otherwise learning rate scheduling would be ineffective."
            )

        data.getitem_size = max(1, round(data_batch_size / config.DATALOADER_FETCHES_PER_BATCH))
        data_train, data_val = data.random_split([1 - val_split, val_split], random_state=random_seed)
        data_train.prepare_shuffle(num_workers=config.ARRAY_SHUFFLE_NUM_WORKERS, random_seed=random_seed)
        data_val.prepare_shuffle(num_workers=config.ARRAY_SHUFFLE_NUM_WORKERS, random_seed=random_seed)

        train_loader = DataLoader(
            data_train, batch_size=config.DATALOADER_FETCHES_PER_BATCH, shuffle=True,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY,
            drop_last=len(data_train) > config.DATALOADER_FETCHES_PER_BATCH,
            generator=torch.Generator().manual_seed(random_seed),
            persistent_workers=False
        )
        val_loader = DataLoader(
            data_val, batch_size=config.DATALOADER_FETCHES_PER_BATCH, shuffle=True,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
            generator=torch.Generator().manual_seed(random_seed),
            persistent_workers=False
        )

        self.align_burnin = align_burnin

        default_plugins = [Tensorboard()]
        if reduce_lr_patience:
            default_plugins.append(LRScheduler(
                self.vae_optim, self.dsc_optim,
                monitor=self.earlystop_loss, patience=reduce_lr_patience,
                burnin=self.align_burnin if safe_burnin else 0
            ))
        if patience:
            default_plugins.append(EarlyStopping(
                monitor=self.earlystop_loss, patience=patience,
                burnin=self.align_burnin if safe_burnin else 0,
                wait_n_lrs=wait_n_lrs or 0
            ))
        plugins = default_plugins + (plugins or [])
        try:
            super().fit(
                train_loader, val_loader=val_loader,
                max_epochs=max_epochs, random_seed=random_seed,
                directory=directory, plugins=plugins
            )
        finally:
            data.clean()
            data_train.clean()
            data_val.clean()
            self.align_burnin = None

    def get_losses(  # pylint: disable=arguments-differ
            self, data: ArrayDataset,
            data_batch_size: int = None,
            random_seed: int = None
    ) -> Mapping[str, float]:
        required_kwargs = ("data_batch_size", "random_seed")
        for required_kwarg in required_kwargs:
            if locals()[required_kwarg] is None:
                raise ValueError(f"`{required_kwarg}` must be specified!")

        data.getitem_size = data_batch_size
        data.prepare_shuffle(num_workers=config.ARRAY_SHUFFLE_NUM_WORKERS, random_seed=random_seed)

        loader = DataLoader(
            data, batch_size=1, shuffle=True, drop_last=False,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY,
            generator=torch.Generator().manual_seed(random_seed),
            persistent_workers=False
        )

        try:
            losses = super().get_losses(loader)
        finally:
            data.clean()

        return losses

    def state_dict(self) -> Mapping[str, Any]:
        return {
            **super().state_dict(),
            "vae_optim": self.vae_optim.state_dict(),
            "dsc_optim": self.dsc_optim.state_dict()
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.vae_optim.load_state_dict(state_dict.pop("vae_optim"))
        self.dsc_optim.load_state_dict(state_dict.pop("dsc_optim"))
        super().load_state_dict(state_dict)
    
    def __repr__(self):
        vae_optim = repr(self.vae_optim).replace("    ", "  ").replace("\n", "\n  ")
        dsc_optim = repr(self.dsc_optim).replace("    ", "  ").replace("\n", "\n  ")
        return (
            f"{type(self).__name__}(\n"
            f"  lam_align: {self.lam_align}\n"
            f"  vae_optim: {vae_optim}\n"
            f"  dsc_optim: {dsc_optim}\n"
            f"  freeze_u: {self.freeze_u}\n"
            f")"
        )
    

#--------------------------------- Public API ----------------------------------
    
@logged
class SCKADEModel(Model):

    """
    Parameters
    ----------
    adatas
        Datasets (indexed by modality name)
    latent_dim
        Latent dimensionality
    h_depth
        Hidden layer depth for encoder and discriminator
    h_dim
        Hidden layer dimensionality for encoder and discriminator
    dropout
        Dropout rate
    shared_batches
        Whether the same batches are shared across modalities
    random_seed
        Random seed
    """

    NET_TYPE = SCKADE
    TRAINER_TYPE = SCKADETrainer

    ALIGN_BURNIN_PRG: float = 8.0  # Effective optimization progress of align_burnin (learning rate * iterations)
    MAX_EPOCHS_PRG: float = 48.0  # Effective optimization progress of max_epochs (learning rate * iterations)
    PATIENCE_PRG: float = 4.0  # Effective optimization progress of patience (learning rate * iterations)
    REDUCE_LR_PATIENCE_PRG: float = 2.0  # Effective optimization progress of reduce_lr_patience (learning rate * iterations)

    def __init__(
            self, adatas: Mapping[str, AnnData],
            latent_dim: int = 50,
            h_depth: int = 2, h_dim: int = 256,
            dropout: float = 0.1, shared_batches: bool = False,
            random_seed: int = 0
    ) -> None:
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)

        self.modalities, x2u, u2x, all_ct = {}, {}, {}, set()
        for k, adata in adatas.items():
            if config.ANNDATA_KEY not in adata.uns:
                raise ValueError(
                    f"The '{k}' dataset has not been configured. "
                    f"Please call `configure_dataset` first!"
                )
            data_config = copy.deepcopy(adata.uns[config.ANNDATA_KEY])
            if data_config["rep_dim"] and data_config["rep_dim"] < latent_dim:
                self.logger.warning(
                    "It is recommended that `use_rep` dimensionality "
                    "be equal or larger than `latent_dim`."
                )

            x2u[k] = Encoder(
                data_config["rep_dim"] or len(data_config["features"]), latent_dim,
                h_depth=h_depth, h_dim=h_dim, dropout=dropout
            )
            data_config["batches"] = pd.Index([]) if data_config["batches"] is None \
                else pd.Index(data_config["batches"])
            u2x[k] = Decoder(
                latent_dim, len(data_config["features"]),
                # latent_dim, data_config["rep_dim"],
                n_batches=max(data_config["batches"].size, 1)
            )

            all_ct = all_ct.union(
                set() if data_config["cell_types"] is None
                else data_config["cell_types"]
            )
            self.modalities[k] = data_config
        all_ct = pd.Index(all_ct).sort_values()
        for modality in self.modalities.values():
            modality["cell_types"] = all_ct
        if shared_batches:
            all_batches = [modality["batches"] for modality in self.modalities.values()]
            ref_batch = all_batches[0]
            for batches in all_batches:
                if not np.array_equal(batches, ref_batch):
                    raise RuntimeError("Batches must match when using `shared_batches`!")
            du_n_batches = ref_batch.size
        else:
            du_n_batches = 0
        du = Discriminator(
            latent_dim, len(self.modalities), n_batches=du_n_batches,
            h_depth=h_depth, h_dim=h_dim, dropout=dropout
        )
        prior = Prior()
        super().__init__(
            x2u, u2x, du, prior,
            u2c=None if all_ct.empty else Classifier(latent_dim, all_ct.size)
        )

    def freeze_cells(self) -> None:
        r"""
        Freeze cell embeddings
        """
        self.trainer.freeze_u = True

    def unfreeze_cells(self) -> None:
        r"""
        Unfreeze cell embeddings
        """
        self.trainer.freeze_u = False

    def adopt_pretrained_model(
            self, source: "SCKADEModel", submodule: Optional[str] = None
    ) -> None:
        """
        Adopt buffers and parameters from a pretrained model

        Parameters
        ----------
        source
            Source model to be adopted
        submodule
            Only adopt a specific submodule (e.g., ``"x2u"``)
        """
        source, target = source.net, self.net
        if submodule:
            source = get_chained_attr(source, submodule)
            target = get_chained_attr(target, submodule)
        for k, t in itertools.chain(target.named_parameters(), target.named_buffers()):
            try:
                s = get_chained_attr(source, k)
            except AttributeError:
                self.logger.warning("Missing: %s", k)
                continue
            if isinstance(t, torch.nn.Parameter):
                t = t.data
            if isinstance(s, torch.nn.Parameter):
                s = s.data
            if s.shape != t.shape:
                self.logger.warning("Shape mismatch: %s", k)
                continue
            s = s.to(device=t.device, dtype=t.dtype)
            t.copy_(s)
            self.logger.debug("Copied: %s", k)

    def compile(  # pylint: disable=arguments-differ
            self, lam_data: float = 1.0,
            lam_kl: float = 1.0,
            lam_align: float = 0.05,
            lam_sup: float = 0.02,
            normalize_u: bool = False,
            modality_weight: Optional[Mapping[str, float]] = None,
            lr: float = 2e-3, **kwargs
    ) -> None:
        """
        Prepare model for training

        Parameters
        ----------
        lam_data
            Data weight
        lam_kl
            KL weight
        lam_align
            Adversarial alignment weight
        lam_sup
            Cell type supervision weight
        normalize_u
            Whether to L2 normalize cell embeddings before decoder
        modality_weight
            Relative modality weight (indexed by modality name)
        lr
            Learning rate
        **kwargs
            Additional keyword arguments passed to trainer
        """
        if modality_weight is None:
            modality_weight = {k: 1.0 for k in self.net.keys}
        super().compile(
            lam_data=lam_data, lam_kl=lam_kl,
            lam_align=lam_align, lam_sup=lam_sup,
            normalize_u=normalize_u, modality_weight=modality_weight,
            optim="RMSprop", lr=lr, **kwargs
        )

    def fit(  # pylint: disable=arguments-differ
            self, adatas: Mapping[str, AnnData],
            val_split: float = 0.1, data_batch_size: int = 128,
            align_burnin: int = AUTO, safe_burnin: bool = True,
            max_epochs: int = AUTO, patience: Optional[int] = AUTO,
            reduce_lr_patience: Optional[int] = AUTO,
            wait_n_lrs: int = 1, directory: Optional[os.PathLike] = None
    ) -> None:
        """
        Fit model on given datasets

        Parameters
        ----------
        adatas
            Datasets (indexed by modality name)
        val_split
            Validation split
        data_batch_size
            Number of cells in each data minibatch
        align_burnin
            Number of epochs to wait before starting alignment
        safe_burnin
            Whether to postpone learning rate scheduling and earlystopping
            until after the burnin stage
        max_epochs
            Maximal number of epochs
        patience
            Patience of early stopping
        reduce_lr_patience
            Patience to reduce learning rate
        wait_n_lrs
            Wait n learning rate scheduling events before starting early stopping
        directory
            Directory to store checkpoints and tensorboard logs
        """
        data = AnnDataset(
            [adatas[key] for key in self.net.keys],
            [self.modalities[key] for key in self.net.keys],
            mode="train"
        )

        batch_per_epoch = data.size * (1 - val_split) / data_batch_size
        if align_burnin == AUTO:
            align_burnin = max(
                ceil(self.ALIGN_BURNIN_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.ALIGN_BURNIN_PRG)
            )
            self.logger.info("Setting `align_burnin` = %d", align_burnin)
        if max_epochs == AUTO:
            max_epochs = max(
                ceil(self.MAX_EPOCHS_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.MAX_EPOCHS_PRG)
            )
            self.logger.info("Setting `max_epochs` = %d", max_epochs)
        if patience == AUTO:
            patience = max(
                ceil(self.PATIENCE_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.PATIENCE_PRG)
            )
            self.logger.info("Setting `patience` = %d", patience)
        if reduce_lr_patience == AUTO:
            reduce_lr_patience = max(
                ceil(self.REDUCE_LR_PATIENCE_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.REDUCE_LR_PATIENCE_PRG)
            )
            self.logger.info("Setting `reduce_lr_patience` = %d", reduce_lr_patience)

        if self.trainer.freeze_u:
            self.logger.info("Cell embeddings are frozen")

        super().fit(
            data, val_split=val_split,
            data_batch_size=data_batch_size,
            align_burnin=align_burnin, safe_burnin=safe_burnin,
            max_epochs=max_epochs, patience=patience,
            reduce_lr_patience=reduce_lr_patience, wait_n_lrs=wait_n_lrs,
            random_seed=self.random_seed,
            directory=directory
        )

    @torch.no_grad()
    def get_losses(  # pylint: disable=arguments-differ
            self, adatas: Mapping[str, AnnData],
            data_batch_size: int = 128,
    ) -> Mapping[str, np.ndarray]:
        """
        Compute loss function values

        Parameters
        ----------
        adatas
            Datasets (indexed by modality name)
        data_batch_size
            Number of cells in each data minibatch
        Returns
        -------
        losses
            Loss function values
        """
        data = AnnDataset(
            [adatas[key] for key in self.net.keys],
            [self.modalities[key] for key in self.net.keys],
            mode="train"
        )
        return super().get_losses(
            data, data_batch_size=data_batch_size,
            random_seed=self.random_seed
        )

    @torch.no_grad()
    def encode_data(
            self, key: str, adata: AnnData, batch_size: int = 128,
            n_sample: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute data (cell) embedding

        Parameters
        ----------
        key
            Modality key
        adata
            Input dataset
        batch_size
            Size of minibatches
        n_sample
            Number of samples from the embedding distribution,
            by default ``None``, returns the mean of the embedding distribution.

        Returns
        -------
        data_embedding
            Data (cell) embedding
            with shape :math:`n_{cell} \times n_{dim}`
            if ``n_sample`` is ``None``,
            or shape :math:`n_{cell} \times n_{sample} \times n_{dim}`
            if ``n_sample`` is not ``None``.
        """
        self.net.eval()
        encoder = self.net.x2u[key]
        data = AnnDataset(
            [adata], [self.modalities[key]],
            mode="eval", getitem_size=batch_size
        )
        data_loader = DataLoader(
            data, batch_size=1, shuffle=False,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
            persistent_workers=False
        )
        result = []
        for x, xrep, *_ in data_loader:
            u = encoder(
                x.to(self.net.device, non_blocking=True),
                xrep.to(self.net.device, non_blocking=True),
                lazy_normalizer=True
            )[0]
            if n_sample:
                result.append(u.sample((n_sample, )).cpu().permute(1, 0, 2))
            else:
                result.append(u.mean.detach().cpu())
        return torch.cat(result).numpy()

    @torch.no_grad()
    def decode_data(
            self, source_key: str, target_key: str,
            adata: AnnData,
            target_libsize: Optional[Union[float, np.ndarray]] = None,
            target_batch: Optional[np.ndarray] = None,
            batch_size: int = 128
    ) -> np.ndarray:
        """
        Decode data

        Parameters
        ----------
        source_key
            Source modality key
        target_key
            Target modality key
        adata
            Source modality data
        graph
            Guidance graph
        target_libsize
            Target modality library size, by default 1.0
        target_batch
            Target modality batch, by default batch 0
        batch_size
            Size of minibatches

        Returns
        -------
        decoded
            Decoded data

        Note
        ----
        This is EXPERIMENTAL!
        """
        l = target_libsize or 1.0
        if not isinstance(l, np.ndarray):
            l = np.asarray(l)
        l = l.squeeze()
        if l.ndim == 0:  # Scalar
            l = l[np.newaxis]
        elif l.ndim > 1:
            raise ValueError("`target_libsize` cannot be >1 dimensional")
        if l.size == 1:
            l = np.repeat(l, adata.shape[0])
        if l.size != adata.shape[0]:
            raise ValueError("`target_libsize` must have the same size as `adata`!")
        l = l.reshape((-1, 1))

        use_batch = self.modalities[target_key]["use_batch"]
        batches = self.modalities[target_key]["batches"]
        if use_batch and target_batch is not None:
            target_batch = np.asarray(target_batch)
            if target_batch.size != adata.shape[0]:
                raise ValueError("`target_batch` must have the same size as `adata`!")
            b = batches.get_indexer(target_batch)
        else:
            b = np.zeros(adata.shape[0], dtype=int)

        net = self.net
        device = net.device
        net.eval()

        u = self.encode_data(source_key, adata, batch_size=batch_size)

        data = ArrayDataset(u, b, l, getitem_size=batch_size)
        data_loader = DataLoader(
            data, batch_size=1, shuffle=False,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
            persistent_workers=False
        )
        decoder = net.u2x[target_key]

        result = []
        for u_, b_, l_ in data_loader:
            u_ = u_.to(device, non_blocking=True)
            b_ = b_.to(device, non_blocking=True)
            l_ = l_.to(device, non_blocking=True)
            result.append(decoder(u_, b_, l_).mean.detach().cpu())
        return torch.cat(result).numpy()

    def upgrade(self) -> None:
        if hasattr(self, "domains"):
            self.logger.warning("Upgrading model generated by older versions...")
            self.modalities = getattr(self, "domains")
            delattr(self, "domains")

    def __repr__(self) -> str:
        return (
            f"SCGLUE model with the following network and trainer:\n\n"
            f"{repr(self.net)}\n\n"
            f"{repr(self.trainer)}\n"
        )