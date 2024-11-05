import torch
import torch.distributions as D
import torch.nn.functional as F

from ..num import EPS


#-------------------------------- Distributions --------------------------------

class MSE(D.Distribution):

    def __init__(self, loc: torch.Tensor) -> None:
        super().__init__(validate_args=False)
        self.loc = loc

    def log_prob(self, value: torch.Tensor) -> None:
        return -F.mse_loss(self.loc, value)

    @property
    def mean(self) -> torch.Tensor:
        return self.loc


class RMSE(MSE):

    def log_prob(self, value: torch.Tensor) -> None:
        return -F.mse_loss(self.loc, value).sqrt()


class ZINB(D.NegativeBinomial):

    def __init__(
            self, zi_logits: torch.Tensor,
            total_count: torch.Tensor, logits: torch.Tensor = None
    ) -> None:
        super().__init__(total_count, logits=logits)
        self.zi_logits = zi_logits

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        raw_log_prob = super().log_prob(value)
        zi_log_prob = torch.empty_like(raw_log_prob)
        z_mask = value.abs() < EPS
        z_zi_logits, nz_zi_logits = self.zi_logits[z_mask], self.zi_logits[~z_mask]
        zi_log_prob[z_mask] = (
            raw_log_prob[z_mask].exp() + z_zi_logits.exp() + EPS
        ).log() - F.softplus(z_zi_logits)
        zi_log_prob[~z_mask] = raw_log_prob[~z_mask] - F.softplus(nz_zi_logits)
        return zi_log_prob