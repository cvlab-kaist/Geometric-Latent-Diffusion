import torch as th
import numpy as np
import logging

import enum

from . import path
from .utils import EasyDict, log_state, mean_flat
from .integrators import ode, sde

# minkyng added import
from einops import rearrange

class ModelType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    NOISE = enum.auto()  # the model predicts epsilon
    SCORE = enum.auto()  # the model predicts \nabla \log p(x)
    VELOCITY = enum.auto()  # the model predicts v(x)

class PathType(enum.Enum):
    """
    Which type of path to use.
    """

    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()

class WeightType(enum.Enum):
    """
    Which type of weighting to use.
    """

    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


def truncated_logitnormal_sample(
    shape, mu, sigma, low=0.0, high=1.0
):
    """
    Samples X in (0,1) with Z = logit(X) ~ Normal(mu, sigma^2), truncated so X in [low, high].
    Works for scalars or tensors mu/sigma/low/high with broadcasting.

    Args:
        shape: output batch shape (e.g., (N,) or (N,M)). Leave () to broadcast to mu.shape.
        mu, sigma: tensors or floats (sigma > 0).
        low, high: truncation bounds in [0,1]. (low can be 0, high can be 1).
        device, dtype: optional overrides.

    Returns:
        Tensor of samples with shape = broadcast(shape, mu.shape, ...)
    """
    mu   = th.as_tensor(mu)
    sigma= th.as_tensor(sigma)
    low  = th.as_tensor(low)
    high = th.as_tensor(high)

    # Map truncation bounds to logit space; handles 0/1 → ±inf automatically.
    z_low  = th.logit(low)   # = -inf if low==0
    z_high = th.logit(high)  # = +inf if high==1

    # Standardize bounds for the base Normal(0,1)
    base = th.distributions.Normal(th.zeros_like(mu), th.ones_like(sigma))
    alpha = (z_low  - mu) / sigma
    beta  = (z_high - mu) / sigma

    # Truncated-normal inverse CDF sampling:
    # U ~ Uniform(Φ(alpha), Φ(beta));  Z = mu + sigma * Φ^{-1}(U);  X = sigmoid(Z)
    cdf_alpha = base.cdf(alpha)
    cdf_beta  = base.cdf(beta)

    # Draw uniforms on the truncated interval
    out_shape = th.broadcast_shapes(shape, mu.shape, sigma.shape, low.shape, high.shape)
    U = th.rand(out_shape, device=mu.device, dtype=mu.dtype)
    U = cdf_alpha + (cdf_beta - cdf_alpha) * U.clamp_(0, 1)

    Z = mu + sigma * base.icdf(U)
    X = th.sigmoid(Z)

    # Numerical safety when low/high are extremely close; clamp back into [low, high].
    return X.clamp(low, high)


class Transport:

    def __init__(
        self,
        *,
        model_type,
        path_type,
        loss_type,
        time_dist_type,
        time_dist_shift,
        train_eps,
        sample_eps,
    ):
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
        }
        self.loss_type = loss_type
        self.model_type = model_type
        self.time_dist_type = time_dist_type
        self.time_dist_shift = time_dist_shift

        assert self.time_dist_shift >= 1.0, "time distribution shift must be >= 1.0."
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps

    def prior_logp(self, z):
        '''
            Standard multivariate normal prior
            Assume z is batched
        '''
        shape = th.tensor(z.size())
        N = th.prod(shape[1:])
        _fn = lambda x: -N / 2. * np.log(2 * np.pi) - th.sum(x ** 2) / 2.
        return th.vmap(_fn)(z)
    

    def check_interval(
        self, 
        train_eps, 
        sample_eps, 
        *, 
        diffusion_form="SBDM",
        sde=False, 
        reverse=False, 
        eval=False,
        last_step_size=0.0,
    ):
        t0 = 0
        t1 = 1 - 1 / 1000
        eps = train_eps if not eval else sample_eps
        if (type(self.path_sampler) in [path.VPCPlan]):

            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]) \
            and (self.model_type != ModelType.VELOCITY or sde): # avoid numerical issue by taking a first semi-implicit step

            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        
        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1


    def sample(self, x1, x0_override=None):
        """Sampling x0 & t based on shape of x1 (if needed)
          Args:
            x1 - data point; [batch, *dim]
            x0_override - optional starting point (e.g., source features + noise).
                          If provided, use this instead of sampling random noise.
        """
        
        if x0_override is not None:
            x0 = x0_override
        else:
            x0 = th.randn_like(x1)
        dist_options = self.time_dist_type.split("_")
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        if dist_options[0] == "uniform":
            t = th.rand((x1.shape[0],)) * (t1 - t0) + t0
        elif dist_options[0] == "logit-normal":
            assert len(dist_options) == 3, "Logit-normal distribution must specify the mean and variance."
            mu, sigma = float(dist_options[1]), float(dist_options[2])
            assert sigma > 0, "Logit-normal distribution must have positive variance."
            t = truncated_logitnormal_sample(
                (x1.shape[0],), mu=mu, sigma=sigma, low=t0, high=t1
            )
        else:
            raise NotImplementedError(f"Unknown time distribution type {self.time_dist_type}")

        t = t.to(x1)

        #sqrt_size_ratio = 1 / self.time_dist_shift # already sqrted
        t = self.time_dist_shift * t / (1 + (self.time_dist_shift - 1) * t)
        return t, x0, x1
    

    def training_losses(
        self, 
        model,  
        x1, 
        total_view, 
        cond_num,
        model_kwargs=None
    ):
        """Loss for training the score model
        Args:
        - model: backbone model; could be score, noise, or velocity
        - x1: datapoint
        - model_kwargs: additional arguments for the model
        """
        if model_kwargs == None:
            model_kwargs = {}
        
        t, x0, x1 = self.sample(x1)
        t, xt, ut = self.path_sampler.plan(t, x0, x1)
        model_output = model(xt, t, **model_kwargs)
        B, *_, C = xt.shape
        assert model_output.size() == (B, *xt.size()[1:-1], C)
        
        terms = {}
        terms['pred'] = model_output
        if self.model_type == ModelType.VELOCITY:
            terms['loss'] = mean_flat(((model_output - ut) ** 2))
        else: 
            _, drift_var = self.path_sampler.compute_drift(xt, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
            if self.loss_type in [WeightType.VELOCITY]:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight = drift_var / (sigma_t ** 2)
            elif self.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError()
            
            if self.model_type == ModelType.NOISE:
                terms['loss'] = mean_flat(weight * ((model_output - x0) ** 2))
            else:
                terms['loss'] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))
                
        return terms
    
    def training_multiview_losses(
        self, 
        model,  
        x1, 
        total_view, 
        cond_num,
        model_kwargs=None,
        x0=None,  # Optional: explicit starting point for flow (e.g., source features + noise)
        t_override=None,  # Optional: fixed timestep (e.g., 0.5 for validation)
    ):
        """Loss for training the score model
        Args:
        - model: backbone model; could be score, noise, or velocity
        - x1: datapoint (target features)
        - model_kwargs: additional arguments for the model
        - x0: optional starting point for the flow. If None, samples from N(0, I).
        - t_override: optional fixed timestep. If not None, use this instead of random sampling.
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        # Extract concat mode flag (use .get to avoid mutating shared model_kwargs)
        is_concat_mode = model_kwargs.get("is_concat_mode", False)
        x1_global = model_kwargs.get("x1_global", None)  # GT global feature for concat mode
        
        # ------------------------------------------------------------
        # 1. Sample init noise x0 and time t
        # ------------------------------------------------------------
        # In concat mode: ODE goes noise → x1_global (global features)
        # x1 (local features) is used only for conditioning
        if is_concat_mode and x1_global is not None:
            x1_target = x1_global  # GT for diffusion
            x1_cond = x1           # Local features for conditioning
        else:
            x1_target = x1
            x1_cond = x1
            
        t, x0, x1_target = self.sample(x1_target, x0_override=x0)  # t: (BV,), x0/x1_target: (BV, ...)
        
        # Override t if specified (e.g., for validation with fixed timestep)
        if t_override is not None:
            t = th.full_like(t, t_override)
        BV = int(x1_target.shape[0])
        C = x1_target.shape[1]
        H, W = x1_target.shape[2], x1_target.shape[3]
        assert BV % total_view == 0
        B = BV // total_view
        assert 1 <= cond_num < total_view, f"cond_num must satisfy 1 <= cond_num < total_view"

        # Broadcast timestep from the first target view
        t_mv = rearrange(t, "(b v) -> b v", v=total_view)
        t_target = t_mv[:, cond_num:cond_num + 1]
        t = t_target.expand(B, total_view).contiguous()
        t = rearrange(t, "b v -> (b v)")

        # ------------------------------------------------------------
        # 2. Compute (xt, ut) - interpolation between noise and x1_target (global)
        # ------------------------------------------------------------
        t, xt, ut = self.path_sampler.plan(t, x0, x1_target)

        # ------------------------------------------------------------
        # 3. Handle Conditional Views
        #    - Concat Mode: Build [cond(ref+zeropad) | noisy] input
        #    - Target-Only Mode: Clamp cond views to x1 (clean GT)
        # ------------------------------------------------------------
        if is_concat_mode:
            # x1_cond is already [ref_clean | zeros] from prepare_data
            # xt is the noisy global features (C channels)
            
            # Concat along channel dimension: [cond | noisy]
            xt = th.cat([x1_cond, xt], dim=1)  # (BV, 2C, H, W)
            
        else:  # not concat mode, target-only mode
            xt  = rearrange(xt,  "(b v) c h w -> b v c h w", v=total_view)
            x1_5d = rearrange(x1_target,  "(b v) c h w -> b v c h w", v=total_view)
            
            target_mask = th.zeros(B, total_view, 1, 1, 1, device=xt.device, dtype=th.bool)
            target_mask[:, cond_num:] = 1 
            
            # Clamp condition views to clean x1
            xt = th.where(target_mask, xt, x1_5d) 
            xt = rearrange(xt, "b v c h w -> (b v) c h w")

        # ------------------------------------------------------------
        # 4. Forward Pass
        # ------------------------------------------------------------
        xt_flat = xt  # already flat
        
        # Minkyung debugging: Remove duplicate 'total_view' from model_kwargs
        if 'total_view' in model_kwargs:
            model_kwargs = {k: v for k, v in model_kwargs.items() if k != 'total_view'}
        
        
        model_output = model(xt_flat, t, total_view, **model_kwargs)
        
        # ------------------------------------------------------------
        # 5. Compute Loss (against x1_target = x1_global in concat mode)
        # ------------------------------------------------------------
        model_output_5d = rearrange(model_output, "(b v) c h w -> b v c h w", v=total_view)
        x1_target_5d    = rearrange(x1_target,    "(b v) c h w -> b v c h w", v=total_view)
        ut_5d           = rearrange(ut,           "(b v) c h w -> b v c h w", v=total_view)
        
        terms = {}
        terms['pred'] = model_output  # keep flat

        if is_concat_mode:
            # Concat Mode Loss: Supervise against x1_global (global features)
            # Network learns: local → global transformation
            
            diff = (model_output_5d - ut_5d) if self.model_type == ModelType.VELOCITY else (model_output_5d - x1_target_5d)
            diff_sq = diff ** 2
            
            # Separate Ref/Tgt Loss
            # ref_loss: first cond_num views
            # tgt_loss: remaining views
            ref_diff_sq = diff_sq[:, :cond_num]
            tgt_diff_sq = diff_sq[:, cond_num:]
            
            terms['loss'] = mean_flat(rearrange(diff_sq, "b v c h w -> (b v) c h w"))
            terms['ref_loss'] = mean_flat(rearrange(ref_diff_sq, "b v c h w -> (b v) c h w"))
            terms['tgt_loss'] = mean_flat(rearrange(tgt_diff_sq, "b v c h w -> (b v) c h w"))
            
        else:
            # Target-Only Loss (Legacy)
            # Slice only target views
            
            x1_tgt   = x1_5d[:, cond_num:] 
            ut_tgt   = ut_5d[:, cond_num:] 
            pred_tgt = model_output_5d[:, cond_num:] 

            x1_loss_flat   = rearrange(x1_tgt,   "b v c h w -> (b v) c h w")
            ut_loss_flat   = rearrange(ut_tgt,   "b v c h w -> (b v) c h w")
            pred_loss_flat = rearrange(pred_tgt, "b v c h w -> (b v) c h w")

            if self.model_type == ModelType.VELOCITY:
                diff = (pred_loss_flat - ut_loss_flat) ** 2
                terms['loss'] = mean_flat(diff)
            else:
                 # For NOISE/SCORE/etc, we need drift/sigma for target views only
                 # This logic mimics the original implementation
                 
                 # Re-flatten xt for comput_drift to ensure shapes match
                 # Original code passed full xt to compute_drift
                 
                _, drift_var = self.path_sampler.compute_drift(xt, t)
                sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
                
                drift_var = drift_var.view(BV, 1, 1, 1)
                sigma_t   = sigma_t.view(BV, 1, 1, 1)

                drift_mv = rearrange(drift_var, "(b v) c h w -> b v c h w", v=total_view)
                sigma_mv = rearrange(sigma_t,   "(b v) c h w -> b v c h w", v=total_view)
                
                drift_tgt = drift_mv[:, cond_num:]
                sigma_tgt = sigma_mv[:, cond_num:]
                
                num_target = total_view - cond_num
                drift_flat = drift_tgt.reshape(B * num_target, 1, 1, 1)
                sigma_flat = sigma_tgt.reshape(B * num_target, 1, 1, 1)

                if self.loss_type in [WeightType.VELOCITY]:
                    weight = (drift_flat / sigma_flat) ** 2
                elif self.loss_type in [WeightType.LIKELIHOOD]:
                    weight = drift_flat / (sigma_flat ** 2)
                elif self.loss_type in [WeightType.NONE]:
                    weight = 1.0
                else:
                    raise NotImplementedError()

                if self.model_type == ModelType.NOISE:
                    diff = (pred_loss_flat - x1_loss_flat) ** 2
                else:
                    diff = (pred_loss_flat * sigma_flat + x1_loss_flat) ** 2

                if isinstance(weight, float) or (not th.is_tensor(weight)):
                    terms['loss'] = mean_flat(diff)
                else:
                    terms['loss'] = mean_flat(weight * diff)

        return terms
    

    def get_drift(
        self
    ):
        """member function for obtaining the drift of the probability flow ODE"""
        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return (-drift_mean + drift_var * model_output) # by change of variable
        
        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return (-drift_mean + drift_var * score)
        
        def velocity_ode(x, t, model, **model_kwargs):
            # Do NOT mutate model_kwargs as it's shared across ODE solver steps
            total_view = model_kwargs.get('total_view')
            cond_num = model_kwargs.get('cond_num')
            pag_scale = model_kwargs.get('pag_scale')
            cfg_scale = model_kwargs.get('cfg_scale')
            is_concat_mode = model_kwargs.get('is_concat_mode')
            ref_cond = model_kwargs.get('ref_cond') # (B, cond_num, C, H, W)
            
            # Create a copy of kwargs for the model call to avoid unexpected side effects
            m_kwargs = {k: v for k, v in model_kwargs.items() 
                       if k not in ['total_view', 'pag_scale', 'pag_layer_idx', 'cfg_scale']}
            
            # 1. Handle PAG and/or CFG (Guidance)
            if not is_concat_mode:
                pag_layer_idx = model_kwargs.get('pag_layer_idx', None)
                pag_enabled = pag_scale is not None and pag_scale > 0
                cfg_enabled = cfg_scale is not None and cfg_scale > 1.0
                
                if pag_enabled and cfg_enabled:
                    # Use both PAG and CFG
                    if total_view is not None:
                        model_output = model.forward_with_pag_and_cfg(
                            x, t, total_view,
                            cfg_scale=cfg_scale, pag_scale=pag_scale,
                            pag_layer_idx=pag_layer_idx, **m_kwargs)
                    else:
                        model_output = model.forward_with_pag_and_cfg(
                            x, t,
                            cfg_scale=cfg_scale, pag_scale=pag_scale,
                            pag_layer_idx=pag_layer_idx, **m_kwargs)
                elif pag_enabled:
                    # Use PAG only
                    if total_view is not None:
                        model_output = model.forward_with_pag(x, t, total_view, pag_scale=pag_scale, pag_layer_idx=pag_layer_idx, **m_kwargs)
                    else:
                        model_output = model.forward_with_pag(x, t, pag_scale=pag_scale, pag_layer_idx=pag_layer_idx, **m_kwargs)
                elif cfg_enabled:
                    # Use CFG only
                    if total_view is not None:
                        model_output = model.forward_with_cfg(x, t, total_view, cfg_scale=cfg_scale, **m_kwargs)
                    else:
                        model_output = model.forward_with_cfg(x, t, cfg_scale=cfg_scale, **m_kwargs)
                else:
                    # Normal forward pass
                    if total_view is not None:
                        model_output = model(x, t, total_view, **m_kwargs)
                    else:
                        model_output = model(x, t, **m_kwargs) # model_output: (B*V, C, H, W)
            elif is_concat_mode:
                # Input x is (BV, 2C, H, W) where first C are cond, second C are noisy
                # We extract the noisy part and concatenate with CLEAN ref_cond for model input
                C = x.shape[1] // 2
                x_noisy = x[:, C:]
                
                # ref_cond should be (BV, C, H, W) - [ref_clean | zeros]
                if ref_cond is None:
                    raise ValueError("ref_cond must be provided in model_kwargs when is_concat_mode is True")
                
                # Ensure ref_cond is flat (BV, C, H, W)
                if ref_cond.ndim == 5:
                    ref_cond = rearrange(ref_cond, "b v c h w -> (b v) c h w")
                
                model_input = th.cat([ref_cond, x_noisy], dim=1)
                
                # Forward pass with 2C input - apply PAG and/or CFG if enabled
                pag_layer_idx = model_kwargs.get('pag_layer_idx', None)
                pag_enabled = pag_scale is not None and pag_scale > 0
                cfg_enabled = cfg_scale is not None and cfg_scale > 1.0
                
                if pag_enabled and cfg_enabled:
                    # Use both PAG and CFG
                    if total_view is not None:
                        model_output = model.forward_with_pag_and_cfg(
                            model_input, t, total_view, 
                            cfg_scale=cfg_scale, pag_scale=pag_scale, 
                            pag_layer_idx=pag_layer_idx, **m_kwargs)
                    else:
                        model_output = model.forward_with_pag_and_cfg(
                            model_input, t, 
                            cfg_scale=cfg_scale, pag_scale=pag_scale,
                            pag_layer_idx=pag_layer_idx, **m_kwargs)
                elif pag_enabled:
                    # Use PAG only
                    if total_view is not None:
                        model_output = model.forward_with_pag(model_input, t, total_view, pag_scale=pag_scale, pag_layer_idx=pag_layer_idx, **m_kwargs)
                    else:
                        model_output = model.forward_with_pag(model_input, t, pag_scale=pag_scale, pag_layer_idx=pag_layer_idx, **m_kwargs)
                elif cfg_enabled:
                    # Use CFG only
                    if total_view is not None:
                        model_output = model.forward_with_cfg(model_input, t, total_view, cfg_scale=cfg_scale, **m_kwargs)
                    else:
                        model_output = model.forward_with_cfg(model_input, t, cfg_scale=cfg_scale, **m_kwargs)
                else:
                    # Normal forward pass
                    if total_view is not None:
                        model_output = model(model_input, t, total_view, **m_kwargs)
                    else:
                        model_output = model(model_input, t, **m_kwargs)

                # model_output (BV, C, H, W)
                # First C channels (static cond) have 0 velocity
                drift_cond = th.zeros_like(model_output)
                model_output = th.cat([drift_cond, model_output], dim=1)

                return model_output
            BV,C,H,W = model_output.shape
            target_view=(total_view-cond_num)
            B=BV//target_view
            zero=th.zeros((B,cond_num,C,H,W)).to(device=model_output.device,dtype=model_output.dtype)
            model_output=th.concat([zero,model_output.view(B,target_view,C,H,W)],dim=1)
            return model_output.view(B*total_view,C,H,W)
            # 3. Mask condition views (Zero velocity/drift for fixed views)
            # This should ONLY be done for "target_only" training (mostly legacy) or if explicitly requested.
            # In Joint ODE training, ref views must float, so we default to False.
            # freeze_cond = model_kwargs.get("freeze_cond", False)
            # if freeze_cond and cond_num is not None and total_view is not None and cond_num > 0:
            #     # model_output: (BV, C, H, W)
            #     # rearrange to (B, V, C, H, W)
            #     mo = rearrange(model_output, "(b v) c h w -> b v c h w", v=total_view)
            #     # Zero out velocity for condition views
            #     mo[:, :cond_num] = 0
            #     # Flatten back
            #     model_output = rearrange(mo, "b v c h w -> (b v) c h w")

            # return model_output

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode
        
        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape, "Output shape from ODE solver must match input shape"
            return model_output

        return body_fn
    

    def get_score(
        self,
    ):
        """member function for obtaining score of 
            x_t = alpha_t * x + sigma_t * eps"""
        if self.model_type == ModelType.NOISE:
            score_fn = lambda x, t, model, **kwargs: model(x, t, **kwargs) / -self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]
        elif self.model_type == ModelType.SCORE:
            score_fn = lambda x, t, model, **kwagrs: model(x, t, **kwagrs)
        elif self.model_type == ModelType.VELOCITY:
            score_fn = lambda x, t, model, **kwargs: self.path_sampler.get_score_from_velocity(model(x, t, **kwargs), x, t)
        else:
            raise NotImplementedError()
        
        return score_fn


class Sampler:
    """Sampler class for the transport model"""
    def __init__(
        self,
        transport,
    ):
        """Constructor for a general sampler; supporting different sampling methods
        Args:
        - transport: an tranport object specify model prediction & interpolant type
        """
        
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()
    
    def __get_sde_diffusion_and_drift(
        self,
        *,
        diffusion_form="SBDM",
        diffusion_norm=1.0,
    ):

        def sde_diffusion_fn(x, t):
            diffusion = self.transport.path_sampler.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm)
            return diffusion
        
        def sde_drift_fn(x, t, model, **kwargs):
            drift_mean = self.drift(x, t, model, **kwargs) - sde_diffusion_fn(x, t) * self.score(x, t, model, **kwargs)
            return drift_mean
    

        return sde_drift_fn, sde_diffusion_fn
    
    def __get_last_step(
        self,
        sde_drift,
        *,
        last_step,
        last_step_size,
    ):
        """Get the last step function of the SDE solver"""
    
        if last_step is None:
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x
        elif last_step == "Mean":
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x - sde_drift(x, t, model, **model_kwargs) * last_step_size
        elif last_step == "Tweedie":
            alpha = self.transport.path_sampler.compute_alpha_t # simple aliasing; the original name was too long
            sigma = self.transport.path_sampler.compute_sigma_t
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x / alpha(t)[0][0] + (sigma(t)[0][0] ** 2) / alpha(t)[0][0] * self.score(x, t, model, **model_kwargs)
        elif last_step == "Euler":
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x - self.drift(x, t, model, **model_kwargs) * last_step_size
        else:
            raise NotImplementedError()

        return last_step_fn

    def sample_sde(
        self,
        *,
        sampling_method="Euler",
        diffusion_form="SBDM",
        diffusion_norm=1.0,
        last_step="Mean",
        last_step_size=0.04,
        num_steps=250,
    ):
        """returns a sampling function with given SDE settings
        Args:
        - sampling_method: type of sampler used in solving the SDE; default to be Euler-Maruyama
        - diffusion_form: function form of diffusion coefficient; default to be matching SBDM
        - diffusion_norm: function magnitude of diffusion coefficient; default to 1
        - last_step: type of the last step; default to identity
        - last_step_size: size of the last step; default to match the stride of 250 steps over [0,1]
        - num_steps: total integration step of SDE
        """

        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = sde(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method,
            time_dist_shift=self.transport.time_dist_shift,
        )

        last_step_fn = self.__get_last_step(sde_drift, last_step=last_step, last_step_size=last_step_size)
            

        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            ts = th.ones(init.size(0), device=init.device) * (1 - t1)
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)

            assert len(xs) == num_steps, "Samples does not match the number of steps"

            return xs

        return _sample
    
    def sample_ode(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
    ):
        """returns a sampling function with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps: 
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        - reverse: whether solving the ODE in reverse (data to noise); default to False
        """
        if reverse:
            drift = lambda x, t, model, **kwargs: self.drift(x, th.ones_like(t) * (1 - t), model, **kwargs)
        else:
            drift = self.drift
        
        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )
        
        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
            time_dist_shift=self.transport.time_dist_shift,
        )
        
        return _ode.sample
    
    # def sample_ode_multiview(
    #     self,
    #     *,
    #     sampling_method="dopri5",
    #     num_steps=50,
    #     atol=1e-6,
    #     rtol=1e-3,
    #     reverse=False,
    # ):
    #     """
    #     returns a sampling function with given ODE settings
    #     """
        
    #     # base drift with optional time flip
    #     if reverse:
    #         base_drift = lambda x, t, model, **kwargs: self.drift(
    #             x, th.ones_like(t) * (1 - t), model, **kwargs
    #         )
    #     else:
    #         base_drift = self.drift

    #     # wrapped drift that applies a target mask if given
    #     def masked_drift(x, t, model, **kwargs):
    #         # compute full drift from model
    #         v = base_drift(x, t, model, **kwargs)  # (BV, C, H, W) typically

    #         target_mask = kwargs.get("target_mask", None)
    #         if target_mask is not None:
    #             # target_mask: (BV, 1, 1, 1) or broadcastable
    #             v = v * target_mask

    #         return v

    #     t0, t1 = self.transport.check_interval(
    #         self.transport.train_eps,
    #         self.transport.sample_eps,
    #         sde=False,
    #         eval=True,
    #         reverse=reverse,
    #         last_step_size=0.0,
    #     )

    #     _ode = ode(
    #         drift=masked_drift,  # use masked drift here
    #         t0=t0,
    #         t1=t1,
    #         sampler_type=sampling_method,
    #         num_steps=num_steps,
    #         atol=atol,
    #         rtol=rtol,
    #         time_dist_shift=self.transport.time_dist_shift,
    #     )
        
    #     return _ode.sample
