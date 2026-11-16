import math
from dataclasses import dataclass

import torch
from diffusers import StableAudioPipeline
from diffusers.models.embeddings import get_1d_rotary_pos_embed

from text_to_music.cfg import cfg

MODEL_ID = "stabilityai/stable-audio-open-1.0"
LATENT_DIMS = (-2, -1)
TRAJECTORY_KEYS = ("difference_norm", "difference_norm_valid", "parallel_fraction", "cond_norm")


@dataclass(frozen=True)
class Guidance:
    mode: str = "cfg"
    scale: float = 7.0
    apg_eta: float = 0.0
    apg_momentum: float = 0.0
    apg_norm_threshold: float = 0.0

    def __post_init__(self):
        assert self.mode in ("cfg", "apg"), self.mode
        assert self.apg_norm_threshold >= 0
        # The paper's buffer adds momentum * running, so its useful range is negative.
        assert -1 < self.apg_momentum <= 0, self.apg_momentum


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


@torch.no_grad()
def generate(pipe, prompt: str, guidance: Guidance, steps: int = 100, seconds: float = 10.0, seed: int = 0):
    device = pipe._execution_device
    sampling_rate = pipe.vae.config.sampling_rate
    scheduler = pipe.scheduler

    prompt_embeds = pipe.encode_prompt(prompt, device, do_classifier_free_guidance=False)
    seconds_start, seconds_end = pipe.encode_duration(0.0, seconds, device, False, batch_size=1)
    condition = torch.cat([prompt_embeds, seconds_start, seconds_end], dim=1)
    encoder_hidden_states = torch.cat([torch.zeros_like(condition), condition])
    global_hidden_states = torch.cat([seconds_start, seconds_end], dim=2).repeat(2, 1, 1)

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
        pipe.rotary_embed_dim, latents.shape[2] + global_hidden_states.shape[1],
        use_real=True, repeat_interleave_real=False,
    )
    buffer = MomentumBuffer(guidance.apg_momentum)
    trajectory = []

    for t in scheduler.timesteps:
        model_input = scheduler.scale_model_input(torch.cat([latents] * 2), t)
        output = pipe.transformer(
            model_input, t.unsqueeze(0),
            encoder_hidden_states=encoder_hidden_states,
            global_hidden_states=global_hidden_states,
            rotary_embedding=rotary_embedding,
            return_dict=False,
        )[0]
        v_uncond, v_cond = output.chunk(2)
        sigma = scheduler.sigmas[scheduler.step_index].float()
        c_skip, c_out = output_coefficients(scheduler, sigma)
        sample = latents.float()
        denoised_cond = c_skip * sample + c_out * v_cond.float()
        denoised_uncond = c_skip * sample + c_out * v_uncond.float()
        trajectory.append(guidance_stats(denoised_cond, denoised_uncond, valid_frames))
        if guidance.mode == "cfg":
            v = cfg(v_uncond, v_cond, guidance.scale)
        else:
            denoised = apg(denoised_cond, denoised_uncond, guidance, buffer)
            v = ((denoised - c_skip * sample) / c_out).to(v_cond.dtype)
        latents = scheduler.step(v, t, latents, **extra_step_kwargs).prev_sample

    audio = pipe.vae.decode(latents).sample
    per_step = torch.stack(trajectory).T.cpu().tolist()
    info = {
        "noise_fingerprint": noise_fingerprint,
        "valid_frames": valid_frames,
        "trajectory": {key: [round(value, 5) for value in values] for key, values in zip(TRAJECTORY_KEYS, per_step)},
    }
    return audio[0, :, : int(seconds * sampling_rate)], info
