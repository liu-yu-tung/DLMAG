import math
from dataclasses import dataclass, fields

import torch
from diffusers import StableAudioPipeline
from diffusers.models.embeddings import get_1d_rotary_pos_embed

from text_to_music.cfg import cfg

MODEL_ID = "stabilityai/stable-audio-open-1.0"
LATENT_DIMS = (-2, -1)
TRAJECTORY_KEYS = ("difference_norm", "difference_norm_valid", "parallel_fraction", "cond_norm")
EXTENSION_FIELDS = ("interval_sigma_low", "interval_sigma_high", "rescale_phi", "second_scale", "rescale_valid_only", "extra_scale")


@dataclass(frozen=True)
class Guidance:
    mode: str = "cfg"
    scale: float = 7.0
    apg_eta: float = 0.0
    apg_momentum: float = 0.0
    apg_norm_threshold: float = 0.0
    interval_sigma_low: float = 0.0
    interval_sigma_high: float = math.inf
    rescale_phi: float = 0.0
    second_scale: float = 0.0
    rescale_valid_only: bool = False
    extra_scale: float = 0.0

    def __post_init__(self):
        assert self.mode in ("cfg", "apg"), self.mode
        assert self.apg_norm_threshold >= 0
        # The paper's buffer adds momentum * running, so its useful range is negative.
        assert -1 < self.apg_momentum <= 0, self.apg_momentum
        assert 0 <= self.interval_sigma_low < self.interval_sigma_high, (self.interval_sigma_low, self.interval_sigma_high)
        assert 0 <= self.rescale_phi <= 1, self.rescale_phi
        assert self.second_scale >= 0, self.second_scale
        assert self.mode == "cfg" or (self.rescale_phi == 0 and self.second_scale == 0), "rescale and second prompt need cfg mode"
        assert self.rescale_phi > 0 or not self.rescale_valid_only, "rescale_valid_only needs rescale_phi"
        assert self.second_scale == 0 or self.extra_scale == 0, "one second prompt: composed share or extra term"
        assert self.mode == "cfg" or self.extra_scale == 0, "the extra prompt term needs cfg mode"

    def settings(self) -> dict:
        # Extension fields are listed only when changed, so gen_info.json of runs from before them still matches on resume.
        defaults = {field.name: field.default for field in fields(self)}
        return {name: getattr(self, name) for name in defaults if name not in EXTENSION_FIELDS or getattr(self, name) != defaults[name]}


def load_pipeline(device: str = "cuda", compile_transformer: bool = False) -> StableAudioPipeline:
    pipe = StableAudioPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)
    if compile_transformer:
        # In place, so pipe.transformer keeps its class and config; Inductor emits Triton kernels.
        pipe.transformer.compile()
    return pipe


class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = None

    def update(self, value: torch.Tensor) -> torch.Tensor:
        if self.running_average is None:
            self.running_average = value
        else:
            self.running_average = value + self.momentum * self.running_average
        return self.running_average


def apg(denoised_cond, denoised_uncond, guidance: Guidance, buffer: MomentumBuffer):
    # Adaptive Projected Guidance (Sadat et al., ICLR 2025, Algorithm 1) on denoised predictions.
    difference = denoised_cond - denoised_uncond
    if guidance.apg_momentum != 0:
        difference = buffer.update(difference)
    if guidance.apg_norm_threshold > 0:
        norm = difference.norm(p=2, dim=LATENT_DIMS, keepdim=True)
        difference = difference * torch.clamp(guidance.apg_norm_threshold / norm, max=1.0)
    unit = denoised_cond / denoised_cond.norm(p=2, dim=LATENT_DIMS, keepdim=True).clamp_min(1e-12)
    parallel = (difference * unit).sum(dim=LATENT_DIMS, keepdim=True) * unit
    orthogonal = difference - parallel
    return denoised_cond + (guidance.scale - 1) * (orthogonal + guidance.apg_eta * parallel)


def output_coefficients(scheduler, sigma):
    # Must mirror CosineDPMSolverMultistepScheduler.precondition_outputs (v-prediction branch).
    sigma_data = scheduler.config.sigma_data
    assert scheduler.config.prediction_type == "v_prediction", scheduler.config.prediction_type
    c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
    c_out = -sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
    return c_skip, c_out


def guidance_stats(denoised_cond, denoised_uncond, valid_frames: int) -> torch.Tensor:
    difference = denoised_cond - denoised_uncond
    difference_norm = difference.norm()
    cond_norm = denoised_cond.norm()
    parallel_fraction = (difference * denoised_cond).sum().abs() / (difference_norm * cond_norm).clamp_min(1e-12)
    return torch.stack([difference_norm, difference[..., :valid_frames].norm(), parallel_fraction, cond_norm])


def rescale_to_conditional(guided, conditional, phi, valid_frames=None):
    # Lin et al., WACV 2024, section 3.4; same as diffusers' rescale_noise_cfg (std over all dims except batch).
    # valid_frames: std over the frames that hold the clip only; the padded tail has no guidance difference and pulls the ratio toward 1.
    dims = list(range(1, guided.ndim))
    measured, reference = (guided, conditional) if valid_frames is None else (guided[..., :valid_frames], conditional[..., :valid_frames])
    rescaled = guided * (reference.std(dim=dims, keepdim=True) / measured.std(dim=dims, keepdim=True))
    return phi * rescaled + (1 - phi) * guided


def composed_cfg(noise_pred_uncond, noise_pred_cond, noise_pred_second, guidance_scale, second_scale):
    # Conjunction of Liu et al., ECCV 2022; guidance_scale is the total weight, second_scale the second prompt's share.
    return (
        noise_pred_uncond
        + (guidance_scale - second_scale) * (noise_pred_cond - noise_pred_uncond)
        + second_scale * (noise_pred_second - noise_pred_uncond)
    )


def extra_term_cfg(noise_pred_uncond, noise_pred_cond, noise_pred_extra, guidance_scale, extra_scale):
    # Negation (extra_scale < 0) and conjunction terms of Liu et al., ECCV 2022, added on top of plain CFG so the caption keeps
    # its full weight; a composed share would raise the caption's weight when the share goes negative (side-agent note 23:50).
    return cfg(noise_pred_uncond, noise_pred_cond, guidance_scale) + extra_scale * (noise_pred_extra - noise_pred_uncond)


def guidance_active(sigma: float, guidance: Guidance) -> bool:
    # Kynkaanniemi et al., NeurIPS 2024: guide only inside (sigma_low, sigma_high]; outside, the conditional prediction alone.
    return guidance.interval_sigma_low < sigma <= guidance.interval_sigma_high


def guided_output(v_uncond, v_cond, v_second, sample, c_skip, c_out, guidance: Guidance, buffer: MomentumBuffer, valid_frames: int):
    if v_uncond is None:
        return v_cond, torch.full((len(TRAJECTORY_KEYS),), math.nan, device=sample.device, dtype=sample.dtype)
    denoised_cond = c_skip * sample + c_out * v_cond.float()
    denoised_uncond = c_skip * sample + c_out * v_uncond.float()
    stats = guidance_stats(denoised_cond, denoised_uncond, valid_frames)
    if guidance.mode == "cfg":
        if v_second is None:
            v = cfg(v_uncond, v_cond, guidance.scale)
        elif guidance.extra_scale != 0:
            v = extra_term_cfg(v_uncond, v_cond, v_second, guidance.scale, guidance.extra_scale)
        else:
            v = composed_cfg(v_uncond, v_cond, v_second, guidance.scale, guidance.second_scale)
        if guidance.rescale_phi > 0:
            v = rescale_to_conditional(v.float(), v_cond.float(), guidance.rescale_phi,
                                       valid_frames if guidance.rescale_valid_only else None).to(v_cond.dtype)
    else:
        denoised = apg(denoised_cond, denoised_uncond, guidance, buffer)
        v = ((denoised - c_skip * sample) / c_out).to(v_cond.dtype)
    return v, stats


def split_output(output: torch.Tensor, active: bool, has_second: bool):
    if not active:
        return None, output, None
    v_uncond, v_cond, *rest = output.chunk(3 if has_second else 2)
    return v_uncond, v_cond, rest[0] if rest else None


@torch.no_grad()
def generate(pipe, prompt: str, guidance: Guidance, steps: int = 100, seconds: float = 10.0, seed: int = 0,
             second_prompt: str | None = None):
    assert (second_prompt is None) == (guidance.second_scale == 0 and guidance.extra_scale == 0), "second prompt and its scale go together"
    device = pipe._execution_device
    sampling_rate = pipe.vae.config.sampling_rate
    scheduler = pipe.scheduler

    prompt_embeds = pipe.encode_prompt(prompt, device, do_classifier_free_guidance=False)
    seconds_start, seconds_end = pipe.encode_duration(0.0, seconds, device, False, batch_size=1)
    condition = torch.cat([prompt_embeds, seconds_start, seconds_end], dim=1)
    conditions = [torch.zeros_like(condition), condition]
    if second_prompt is not None:
        second_embeds = pipe.encode_prompt(second_prompt, device, do_classifier_free_guidance=False)
        conditions.append(torch.cat([second_embeds, seconds_start, seconds_end], dim=1))
    global_condition = torch.cat([seconds_start, seconds_end], dim=2)
    branches = {
        True: (torch.cat(conditions), global_condition.repeat(len(conditions), 1, 1)),
        False: (condition, global_condition),
    }

    generator = torch.Generator(device).manual_seed(seed)
    scheduler.set_timesteps(steps, device=device)
    latents = pipe.prepare_latents(
        1, pipe.transformer.config.in_channels, int(pipe.transformer.config.sample_size),
        condition.dtype, device, generator, audio_channels=pipe.vae.config.audio_channels,
    )
    noise_fingerprint = float(latents.double().sum())
    # SAO always samples 1024 latent frames (47.5 s); only the first ones carry the requested seconds.
    valid_frames = math.ceil(seconds * sampling_rate / math.prod(pipe.vae.config.downsampling_ratios))
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, 0.0)
    rotary_embedding = get_1d_rotary_pos_embed(
        pipe.rotary_embed_dim, latents.shape[2] + global_condition.shape[1],
        use_real=True, repeat_interleave_real=False,
    )
    buffer = MomentumBuffer(guidance.apg_momentum)
    trajectory = []

    forward_samples = 0

    for t in scheduler.timesteps:
        scaled_latents = scheduler.scale_model_input(latents, t)
        sigma = scheduler.sigmas[scheduler.step_index].float()
        active = guidance_active(float(sigma), guidance)
        encoder_hidden_states, global_hidden_states = branches[active]
        batch = encoder_hidden_states.shape[0]
        forward_samples += batch
        output = pipe.transformer(
            torch.cat([scaled_latents] * batch), t.unsqueeze(0),
            encoder_hidden_states=encoder_hidden_states,
            global_hidden_states=global_hidden_states,
            rotary_embedding=rotary_embedding,
            return_dict=False,
        )[0]
        v_uncond, v_cond, v_second = split_output(output, active, second_prompt is not None)
        c_skip, c_out = output_coefficients(scheduler, sigma)
        v, stats = guided_output(v_uncond, v_cond, v_second, latents.float(), c_skip, c_out, guidance, buffer, valid_frames)
        trajectory.append(stats)
        latents = scheduler.step(v, t, latents, **extra_step_kwargs).prev_sample

    audio = pipe.vae.decode(latents).sample
    per_step = torch.stack(trajectory).T.cpu().tolist()
    info = {
        "noise_fingerprint": noise_fingerprint,
        "valid_frames": valid_frames,
        "forward_samples": forward_samples,
        "trajectory": {key: [None if math.isnan(value) else round(value, 5) for value in values]
                       for key, values in zip(TRAJECTORY_KEYS, per_step)},
    }
    return audio[0, :, : int(seconds * sampling_rate)], info
