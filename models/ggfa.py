# Gaussian-Gated Feature Adaptation (GGFA) for TIGR-Seg.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class GaussianGatedFeatureAdaptation(nn.Module):
    """
    GGFA for one frozen Q/K/V projection.

    Y0   = X W0^T + b0
    h    = GELU(LN(mean_i(X_i W_d)))
    Ymod = gamma * Y0 + beta
    G_i  = a exp[-0.5 (t_i-mu)^T Sigma^{-1}(t_i-mu)]
    Y    = Y0 + lambda G Ymod
    """

    def __init__(
        self,
        frozen_projection,
        bottleneck_dim=4,
        sigma_min=0.08,
        sigma_max=0.60,
        rho_max=0.90,
        lambda_init=1.0,
    ):
        super().__init__()

        if not (0.0 < sigma_min < sigma_max):
            raise ValueError("Expected 0 < sigma_min < sigma_max.")
        if not (0.0 < rho_max < 1.0):
            raise ValueError("Expected 0 < rho_max < 1.")

        d_in = frozen_projection.in_features
        d_out = frozen_projection.out_features
        r = int(bottleneck_dim)

        self.d_in = d_in
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho_max = float(rho_max)

        # Frozen pretrained projection.
        self.weight = nn.Parameter(
            frozen_projection.weight.detach().clone(),
            requires_grad=False,
        )
        self.bias = (
            None
            if frozen_projection.bias is None
            else nn.Parameter(
                frozen_projection.bias.detach().clone(),
                requires_grad=False,
            )
        )

        # Shared window context used by both adaptation branches.
        self.context_projection = nn.Linear(d_in, r, bias=False)
        self.context_norm = nn.LayerNorm(r)

        # Context-conditioned feature modulation.
        self.gamma_head = nn.Linear(r, d_out)
        self.beta_head = nn.Linear(r, d_out)

        # Predict the six parameters of the correlated Gaussian gate.
        self.gaussian_predictor = nn.Sequential(
            nn.Linear(r, r),
            nn.GELU(),
            nn.Linear(r, 6),
        )

        # Zero initialization preserves the pretrained projection.
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

        self.lambda_residual = nn.Parameter(
            torch.tensor(float(lambda_init))
        )

    @staticmethod
    def _coordinates(num_tokens, device, dtype):
        side = int(math.isqrt(num_tokens))
        if side * side != num_tokens:
            raise ValueError(
                f"Expected N_w=M^2 tokens, got N_w={num_tokens}."
            )

        coord_y, coord_x = torch.meshgrid(
            torch.linspace(0.0, 1.0, side, device=device, dtype=dtype),
            torch.linspace(0.0, 1.0, side, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack((coord_x, coord_y), dim=-1).reshape(
            1, num_tokens, 2
        )

    def _gaussian_gate(self, h, num_tokens):
        raw = self.gaussian_predictor(h)

        mu_x = torch.sigmoid(raw[:, 0])
        mu_y = torch.sigmoid(raw[:, 1])
        sigma_x = self.sigma_min + (
            self.sigma_max - self.sigma_min
        ) * torch.sigmoid(raw[:, 2])
        sigma_y = self.sigma_min + (
            self.sigma_max - self.sigma_min
        ) * torch.sigmoid(raw[:, 3])
        rho = self.rho_max * torch.tanh(raw[:, 4])
        amplitude = torch.sigmoid(raw[:, 5])

        coords = self._coordinates(num_tokens, h.device, h.dtype)
        coord_x = coords[..., 0]
        coord_y = coords[..., 1]

        mu_x = mu_x[:, None]
        mu_y = mu_y[:, None]
        sigma_x = sigma_x[:, None]
        sigma_y = sigma_y[:, None]
        rho = rho[:, None]
        amplitude = amplitude[:, None]

        dx = (coord_x - mu_x) / sigma_x
        dy = (coord_y - mu_y) / sigma_y

        # Closed-form Mahalanobis distance for the correlated Gaussian.
        mahalanobis_sq = (
            dx.square()
            - 2.0 * rho * dx * dy
            + dy.square()
        ) / (1.0 - rho.square())

        return (
            amplitude
            * torch.exp(-0.5 * mahalanobis_sq)
        ).unsqueeze(-1)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.d_in:
            raise ValueError(
                f"Expected [B_w,N_w,{self.d_in}], got {tuple(x.shape)}."
            )

        # Frozen pretrained response.
        y0 = F.linear(x, self.weight, self.bias)

        # Shared window context for modulation and spatial gating.
        h = self.context_projection(x).mean(dim=1)
        h = F.gelu(self.context_norm(h))

        # Context-conditioned feature modulation.
        gamma = torch.tanh(self.gamma_head(h))[:, None, :]
        beta = torch.tanh(self.beta_head(h))[:, None, :]
        y_mod = gamma * y0 + beta

        # Dynamic correlated Gaussian gate.
        gate = self._gaussian_gate(h, x.shape[1])

        # Residual adaptation.
        return y0 + self.lambda_residual * gate * y_mod

def inject_ggfa_into_swin_qkv(
    module,
    bottleneck_dim=4,
    sigma_min=0.08,
    sigma_max=0.60,
    rho_max=0.90,
    lambda_init=1.0,
):
    """Replace only Swin attention query/key/value projections with GGFA."""
    replaced = 0
    is_attention = "attention" in module.__class__.__name__.lower()

    for name, child in list(module.named_children()):
        if isinstance(child, GaussianGatedFeatureAdaptation):
            continue

        if (
            is_attention
            and isinstance(child, nn.Linear)
            and name.lower() in {"query", "key", "value"}
        ):
            setattr(
                module,
                name,
                GaussianGatedFeatureAdaptation(
                    child,
                    bottleneck_dim=bottleneck_dim,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    rho_max=rho_max,
                    lambda_init=lambda_init,
                ),
            )
            replaced += 1
        else:
            replaced += inject_ggfa_into_swin_qkv(
                child,
                bottleneck_dim,
                sigma_min,
                sigma_max,
                rho_max,
                lambda_init,
            )

    return replaced


class SwinV2TinyGGFAEncoder(nn.Module):
    """
    Frozen Swin-V2-Tiny with GGFA in Q/K/V projections.

    Returns V4, V8, V16, V32 with channels (96, 192, 384, 768).
    """

    def __init__(
        self,
        model_name="microsoft/swinv2-tiny-patch4-window8-256",
        bottleneck_dim=4,
        sigma_min=0.08,
        sigma_max=0.60,
        rho_max=0.90,
        lambda_init=1.0,
    ):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name)

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        n_replaced = inject_ggfa_into_swin_qkv(
            self.backbone,
            bottleneck_dim=bottleneck_dim,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho_max=rho_max,
            lambda_init=lambda_init,
        )
        if n_replaced == 0:
            raise RuntimeError(
                "No Swin query/key/value projections were found."
            )

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                "Expected a three-channel BCHW image tensor."
            )

        outputs = self.backbone(
            pixel_values=image,
            output_hidden_states=True,
            return_dict=True,
        )
        features = outputs.reshaped_hidden_states

        if features is None or len(features) < 5:
            raise RuntimeError(
                "Expected hierarchical Swin-V2 hidden states."
            )

        # Hierarchical Swin-V2 features at output strides 4, 8, 16, and 32.
        v4 = features[0]
        v8 = features[1]
        v16 = features[2]
        v32 = features[-1]

        if (
            v4.shape[1],
            v8.shape[1],
            v16.shape[1],
            v32.shape[1],
        ) != (96, 192, 384, 768):
            raise RuntimeError(
                "Unexpected Swin-V2 feature dimensions."
            )

        return v4, v8, v16, v32
