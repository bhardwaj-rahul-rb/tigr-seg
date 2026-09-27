# Text-Initialized Gaussian Refinement (TIGR) Decoder for TIGR-Seg.
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableRefinement(nn.Module):
    """Depthwise 3x3 -> pointwise 1x1 -> GroupNorm -> GELU."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels, channels, 3,
                padding=1, groups=channels, bias=False,
            ),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class TIGRDecoderStage(nn.Module):
    """One TIGR decoder stage at output stride s."""

    def __init__(
        self,
        in_channels,
        skip_channels,
        channels,
        sigma_min=0.04,
        sigma_max=0.50,
        rho_max=0.95,
        correction_bounds=(1.5, 1.5, 1.0, 1.0, 1.0),
        eta_init=0.1,
    ):
        super().__init__()

        if not (0.0 < sigma_min < sigma_max):
            raise ValueError("Expected 0 < sigma_min < sigma_max.")
        if not (0.0 < rho_max < 1.0):
            raise ValueError("Expected 0 < rho_max < 1.")
        if len(correction_bounds) != 5:
            raise ValueError("correction_bounds must contain five values.")

        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho_max = float(rho_max)

        self.register_buffer(
            "correction_bounds",
            torch.tensor(correction_bounds, dtype=torch.float32),
            persistent=False,
        )

        # Multi-scale feature fusion.
        # phi_s and psi_s share the architecture but use separate parameters.
        self.phi = nn.Sequential(
            nn.Conv2d(
                in_channels + skip_channels,
                channels, 1, bias=False,
            ),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            DepthwiseSeparableRefinement(channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(
                channels + 1,
                channels, 1, bias=False,
            ),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            DepthwiseSeparableRefinement(channels),
        )

        # Text-initialized Gaussian and presence proposals.
        self.gaussian_proposal = nn.Linear(channels, 5)
        self.presence_proposal = nn.Linear(channels, 1)

        # Image-guided Gaussian correction heads.
        joint_dim = 3 * channels
        hidden_dim = max(16, channels // 2)

        self.spatial_correction = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 5),
        )
        self.reliability_head = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.presence_correction = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize image-guided corrections to zero so each stage starts from the text proposal.
        nn.init.zeros_(self.spatial_correction[-1].weight)
        nn.init.zeros_(self.spatial_correction[-1].bias)
        nn.init.zeros_(self.presence_correction[-1].weight)
        nn.init.zeros_(self.presence_correction[-1].bias)

        # Gaussian-guided feature reconstruction.
        self.stage_embedding = nn.Parameter(torch.zeros(channels))
        self.reconstruction_mlp = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.reconstruction_projection = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )

        # Text-conditioned channel residual.
        self.gamma_head = nn.Linear(channels, channels)
        self.beta_head = nn.Linear(channels, channels)

        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

        # Gated residual decoder update.
        self.update_gate = nn.Conv2d(3, 1, 1)
        self.eta = nn.Parameter(torch.tensor(float(eta_init)))
        self.refine = DepthwiseSeparableRefinement(channels)

    @staticmethod
    def _previous_prior(previous_prior, feature):
        """Return Up(P_2s); P_32 is zero at the first stage."""
        b, _, h, w = feature.shape

        if previous_prior is None:
            return torch.zeros(
                b, 1, h, w,
                device=feature.device,
                dtype=feature.dtype,
            )

        if previous_prior.shape[-2:] != (h, w):
            previous_prior = F.interpolate(
                previous_prior,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )

        return previous_prior.to(dtype=feature.dtype)

    def _make_gaussian_prior(self, theta, presence, height, width):
        """Eqs. (23)-(26): construct P_s(t) = p_s G_s(t)."""
        mu_x = torch.sigmoid(theta[:, 0])
        mu_y = torch.sigmoid(theta[:, 1])

        sigma_x = self.sigma_min + (
            self.sigma_max - self.sigma_min
        ) * torch.sigmoid(theta[:, 2])

        sigma_y = self.sigma_min + (
            self.sigma_max - self.sigma_min
        ) * torch.sigmoid(theta[:, 3])

        rho = self.rho_max * torch.tanh(theta[:, 4])

        coord_y, coord_x = torch.meshgrid(
            torch.linspace(
                0.0, 1.0, height,
                device=theta.device, dtype=theta.dtype,
            ),
            torch.linspace(
                0.0, 1.0, width,
                device=theta.device, dtype=theta.dtype,
            ),
            indexing="ij",
        )

        coord_x = coord_x[None]
        coord_y = coord_y[None]

        mu_x = mu_x[:, None, None]
        mu_y = mu_y[:, None, None]
        sigma_x = sigma_x[:, None, None]
        sigma_y = sigma_y[:, None, None]
        rho = rho[:, None, None]

        dx = (coord_x - mu_x) / sigma_x
        dy = (coord_y - mu_y) / sigma_y

        mahalanobis_sq = (
            dx.square()
            - 2.0 * rho * dx * dy
            + dy.square()
        ) / (1.0 - rho.square())

        gaussian = torch.exp(
            -0.5 * mahalanobis_sq
        ).unsqueeze(1)

        return presence[:, :, None, None] * gaussian

    def forward(
        self,
        decoder_feature,
        skip_feature,
        text_embedding,
        previous_prior=None,
    ):
        # Multi-scale feature fusion.
        decoder_up = F.interpolate(
            decoder_feature,
            size=skip_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        f0 = self.phi(
            torch.cat([decoder_up, skip_feature], dim=1)
        )

        previous_prior = self._previous_prior(
            previous_prior, f0
        )

        feature = self.psi(
            torch.cat([f0, previous_prior], dim=1)
        )

        b, c, h, w = feature.shape

        # Text-guided Gaussian proposal.
        theta0 = self.gaussian_proposal(text_embedding)
        q0 = self.presence_proposal(text_embedding)

        # Image-guided Gaussian correction.
        joint = torch.cat(
            [
                F.adaptive_avg_pool2d(feature, 1).flatten(1),
                F.adaptive_max_pool2d(feature, 1).flatten(1),
                text_embedding,
            ],
            dim=1,
        )

        bounds = self.correction_bounds.to(
            dtype=feature.dtype
        )

        delta = (
            torch.tanh(self.spatial_correction(joint))
            * bounds[None]
        )

        reliability = torch.sigmoid(
            self.reliability_head(joint)
        )

        theta = theta0 + reliability * delta

        presence_delta = 2.0 * torch.tanh(
            self.presence_correction(joint)
        )

        q = q0 + reliability * presence_delta
        presence = torch.sigmoid(q)

        # Correlated Gaussian prior
        prior = self._make_gaussian_prior(
            theta, presence, h, w
        )

        # Gaussian-guided feature reconstruction
        numerator = (prior * feature).sum(dim=(2, 3))
        denominator = prior.sum(dim=(2, 3))

        # Guard against division by zero for a vanishing prior.
        visual_summary = numerator / denominator.clamp_min(1e-8)

        stage_text = (
            text_embedding
            + self.stage_embedding[None]
        )

        reconstruction_vector = self.reconstruction_mlp(
            torch.cat(
                [visual_summary, stage_text],
                dim=1,
            )
        )

        reconstruction = (
            prior
            * reconstruction_vector[:, :, None, None]
        )
        reconstruction = self.reconstruction_projection(
            reconstruction
        )

        # Text-conditioned channel residual
        gamma = torch.tanh(
            self.gamma_head(text_embedding)
        )[:, :, None, None]

        beta = torch.tanh(
            self.beta_head(text_embedding)
        )[:, :, None, None]

        text_residual = gamma * feature + beta

        # Spatial gate and residual decoder update
        reliability_map = reliability[
            :, :, None, None
        ].expand(b, 1, h, w)

        gate = torch.sigmoid(
            self.update_gate(
                torch.cat(
                    [prior, previous_prior, reliability_map],
                    dim=1,
                )
            )
        )

        updated = (
            feature
            + self.eta
            * gate
            * (reconstruction + text_residual)
        )

        decoder_output = self.refine(updated)

        aux = {
            "prior": prior,
            "theta0": theta0,
            "theta": theta,
            "presence": presence,
            "reliability": reliability,
            "update_gate": gate,
        }

        return decoder_output, prior, aux


class TIGRDecoder(nn.Module):
    """Three-stage TIGR Decoder at output strides 16, 8, and 4."""

    def __init__(
        self,
        text_dim=768,
        sigma_min=0.04,
        sigma_max=0.50,
        rho_max=0.95,
        correction_bounds=(1.5, 1.5, 1.0, 1.0, 1.0),
        eta_init=0.1,
    ):
        super().__init__()

        # Stage-specific text projections tau_s.
        self.text_projection_16 = nn.Linear(text_dim, 384)
        self.text_projection_8 = nn.Linear(text_dim, 192)
        self.text_projection_4 = nn.Linear(text_dim, 96)

        common = dict(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho_max=rho_max,
            correction_bounds=correction_bounds,
            eta_init=eta_init,
        )

        self.stage16 = TIGRDecoderStage(
            768, 384, 384, **common
        )
        self.stage8 = TIGRDecoderStage(
            384, 192, 192, **common
        )
        self.stage4 = TIGRDecoderStage(
            192, 96, 96, **common
        )

    @staticmethod
    def masked_mean(text_tokens, attention_mask):
        """Eq. (3): masked mean of the frozen ClinicalBERT token features."""
        mask = attention_mask.unsqueeze(-1).to(
            dtype=text_tokens.dtype
        )

        return (
            (text_tokens * mask).sum(dim=1)
            / mask.sum(dim=1).clamp_min(1.0)
        )

    def forward(
        self,
        v4,
        v8,
        v16,
        v32,
        text_tokens,
        attention_mask,
    ):
        text_vector = self.masked_mean(
            text_tokens, attention_mask
        )

        z16 = self.text_projection_16(text_vector)
        z8 = self.text_projection_8(text_vector)
        z4 = self.text_projection_4(text_vector)

        # Initialize the decoder with D32 = V32 and P32 = 0.
        d16, p16, aux16 = self.stage16(
            v32, v16, z16, None
        )

        d8, p8, aux8 = self.stage8(
            d16, v8, z8, p16
        )

        d4, p4, aux4 = self.stage4(
            d8, v4, z4, p8
        )

        aux = {
            "stage16": aux16,
            "stage8": aux8,
            "stage4": aux4,
            "priors": (p16, p8, p4),
        }

        return d4, aux


class SegmentationHead(nn.Module):
    """Sub-pixel x4 upsampling followed by a 1x1 segmentation head."""

    def __init__(self, in_channels=96):
        super().__init__()

        self.subpixel = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels * 16,
                kernel_size=1,
            ),
            nn.PixelShuffle(4),
        )

        self.logit_head = nn.Conv2d(
            in_channels, 1, kernel_size=1
        )

    def forward(self, d4):
        logits = self.logit_head(
            self.subpixel(d4)
        )
        
        return torch.sigmoid(logits)
