import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

from text_to_music.cfg import cfg
from text_to_music.sampler import LATENT_DIMS, TRAJECTORY_KEYS, Guidance, MomentumBuffer, apg, guidance_stats, output_coefficients

SHAPE = (1, 64, 1024)
SCALES = (1.0, 3.0, 7.0, 14.0)
SCHEDULER_CONFIGS = sorted(
    (Path.home() / ".cache/huggingface/hub/models--stabilityai--stable-audio-open-1.0/snapshots").glob(
        "*/scheduler/scheduler_config.json"
    )
)
# Fallback values are the diffusers 0.41.0 defaults of CosineDPMSolverMultistepScheduler.
FALLBACK_CONFIG = {"sigma_data": 1.0, "sigma_min": 0.3, "sigma_max": 500.0}


def scheduler_config() -> tuple[dict, str]:
    if SCHEDULER_CONFIGS:
        config = json.loads(SCHEDULER_CONFIGS[-1].read_text())
        return {key: float(config[key]) for key in FALLBACK_CONFIG}, str(SCHEDULER_CONFIGS[-1])
    return dict(FALLBACK_CONFIG), "fallback (model scheduler_config.json not in the Hugging Face cache)"


CONFIG, CONFIG_SOURCE = scheduler_config()
SCHEDULER = SimpleNamespace(config=SimpleNamespace(sigma_data=CONFIG["sigma_data"], prediction_type="v_prediction"))


def sigmas(dtype, count: int = 24) -> torch.Tensor:
    low, high = torch.tensor([CONFIG["sigma_min"], CONFIG["sigma_max"]], dtype=torch.float64).log10()
    return torch.logspace(float(low), float(high), count, dtype=dtype)


def random_tensors(count: int, dtype, seed: int = 0) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(SHAPE, generator=generator, dtype=dtype) for _ in range(count)]


def relative_error(value: torch.Tensor, reference: torch.Tensor) -> float:
    return float((value - reference).norm() / reference.norm())


def inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=LATENT_DIMS)


def norm(a: torch.Tensor) -> torch.Tensor:
    return a.norm(p=2, dim=LATENT_DIMS)


def test_v_to_denoised_round_trip():
    for dtype, tolerance in ((torch.float32, 1e-5), (torch.float64, 1e-12)):
        sample, v = random_tensors(2, dtype)
        for sigma in sigmas(dtype):
            # The sampler scales the latent by sigma before the model call, so test at that magnitude too.
            for scaled_sample in (sample, sample * sigma):
                c_skip, c_out = output_coefficients(SCHEDULER, sigma)
                denoised = c_skip * scaled_sample + c_out * v
                recovered = (denoised - c_skip * scaled_sample) / c_out
                error = relative_error(recovered, v)
                assert error < tolerance, (dtype, float(sigma), error)


def test_output_coefficients_reject_other_prediction_types():
    scheduler = SimpleNamespace(config=SimpleNamespace(sigma_data=1.0, prediction_type="epsilon"))
    try:
        output_coefficients(scheduler, torch.tensor(1.0))
    except AssertionError:
        return
    raise AssertionError("epsilon prediction was accepted")


def test_apg_with_eta_one_equals_cfg():
    sample, v_uncond, v_cond = random_tensors(3, torch.float64)
    for sigma in sigmas(torch.float64, count=8):
        c_skip, c_out = output_coefficients(SCHEDULER, sigma)
        denoised_cond = c_skip * sample + c_out * v_cond
        denoised_uncond = c_skip * sample + c_out * v_uncond
        for scale in SCALES:
            guidance = Guidance("apg", scale, apg_eta=1.0, apg_momentum=0.0, apg_norm_threshold=0.0)
            denoised = apg(denoised_cond, denoised_uncond, guidance, MomentumBuffer(0.0))
            expected = denoised_cond + (scale - 1) * (denoised_cond - denoised_uncond)
            assert relative_error(denoised, expected) < 1e-5, (float(sigma), scale)
            v = (denoised - c_skip * sample) / c_out
            error = relative_error(v, cfg(v_uncond, v_cond, scale))
            assert error < 1e-5, (float(sigma), scale, error)


def test_apg_with_eta_zero_is_orthogonal_to_cond():
    for dtype, tolerance in ((torch.float32, 1e-4), (torch.float64, 1e-10)):
        denoised_cond, denoised_uncond = random_tensors(2, dtype, seed=1)
        # A shared component makes the parallel part large, as in real denoised predictions.
        denoised_uncond = denoised_uncond + 0.8 * denoised_cond
        for threshold in (0.0, 5.0):
            guidance = Guidance("apg", 7.0, apg_eta=0.0, apg_norm_threshold=threshold)
            update = apg(denoised_cond, denoised_uncond, guidance, MomentumBuffer(0.0)) - denoised_cond
            assert float(norm(update).min()) > 0
            cosine = inner(update, denoised_cond) / (norm(update) * norm(denoised_cond))
            assert float(cosine.abs().max()) < tolerance, (dtype, threshold, cosine)


def test_norm_threshold_caps_the_difference():
    denoised_cond, denoised_uncond = random_tensors(2, torch.float64, seed=2)
    difference = denoised_cond - denoised_uncond
    difference_norm = float(norm(difference))
    scale = 7.0
    for threshold in (0.1 * difference_norm, 0.5 * difference_norm):
        guidance = Guidance("apg", scale, apg_eta=1.0, apg_norm_threshold=threshold)
        update = apg(denoised_cond, denoised_uncond, guidance, MomentumBuffer(0.0)) - denoised_cond
        rescaled_norm = float(norm(update)) / (scale - 1)
        assert rescaled_norm <= threshold * (1 + 1e-9), (rescaled_norm, threshold)
        assert abs(rescaled_norm - threshold) < 1e-9 * threshold
        assert relative_error(update / (scale - 1), difference * threshold / difference_norm) < 1e-9
    for threshold in (1.01 * difference_norm, 10 * difference_norm):
        guidance = Guidance("apg", scale, apg_eta=1.0, apg_norm_threshold=threshold)
        update = apg(denoised_cond, denoised_uncond, guidance, MomentumBuffer(0.0)) - denoised_cond
        assert relative_error(update / (scale - 1), difference) < 1e-12, threshold


def test_momentum_buffer_converges_below_the_input():
    (difference,) = random_tensors(1, torch.float64, seed=3)
    for momentum in (-0.25, -0.5, -0.75):
        buffer = MomentumBuffer(momentum)
        for _ in range(400):
            running_average = buffer.update(difference)
        limit = difference / (1 - momentum)
        assert relative_error(running_average, limit) < 1e-9, momentum
        # Negative momentum shrinks a steady difference, so the effective scale is (w - 1) / (1 - momentum).
        assert float(norm(running_average)) < float(norm(difference))


def test_momentum_is_applied_inside_apg():
    denoised_cond, denoised_uncond = random_tensors(2, torch.float64, seed=4)
    momentum, scale = -0.5, 7.0
    guidance = Guidance("apg", scale, apg_eta=1.0, apg_momentum=momentum)
    buffer = MomentumBuffer(momentum)
    for _ in range(200):
        denoised = apg(denoised_cond, denoised_uncond, guidance, buffer)
    expected = denoised_cond + (scale - 1) / (1 - momentum) * (denoised_cond - denoised_uncond)
    assert relative_error(denoised, expected) < 1e-9


def test_guidance_rejects_bad_settings():
    bad_settings = (
        {"mode": "x"},
        {"mode": "apg", "apg_norm_threshold": -1.0},
        {"mode": "apg", "apg_momentum": 0.5},
        {"mode": "apg", "apg_momentum": -1.0},
    )
    for settings in bad_settings:
        try:
            Guidance(**settings)
        except AssertionError:
            continue
        raise AssertionError(f"Guidance accepted {settings}")
    Guidance("cfg", 7.0)
    Guidance("apg", 7.0, apg_eta=0.0, apg_momentum=-0.5, apg_norm_threshold=2.5)


def test_guidance_stats():
    denoised_cond, other = random_tensors(2, torch.float64, seed=5)
    valid_frames = 215

    stats = guidance_stats(denoised_cond, 0.5 * denoised_cond, valid_frames)
    assert stats.shape == (len(TRAJECTORY_KEYS),) == (4,)
    values = dict(zip(TRAJECTORY_KEYS, stats.tolist()))
    assert abs(values["parallel_fraction"] - 1) < 1e-9
    assert abs(values["difference_norm"] - 0.5 * float(denoised_cond.norm())) < 1e-9
    assert abs(values["difference_norm_valid"] - 0.5 * float(denoised_cond[..., :valid_frames].norm())) < 1e-9
    assert abs(values["cond_norm"] - float(denoised_cond.norm())) < 1e-9
    assert values["difference_norm_valid"] < values["difference_norm"]

    orthogonal = other - (other * denoised_cond).sum() / denoised_cond.pow(2).sum() * denoised_cond
    values = dict(zip(TRAJECTORY_KEYS, guidance_stats(denoised_cond, denoised_cond - orthogonal, valid_frames).tolist()))
    assert values["parallel_fraction"] < 1e-9
    assert abs(values["difference_norm"] - float(orthogonal.norm())) < 1e-9

    full = guidance_stats(denoised_cond, other, SHAPE[-1])
    assert float(full[0]) == float(full[1])


if __name__ == "__main__":
    torch.manual_seed(0)
    print(f"scheduler config: {CONFIG} from {CONFIG_SOURCE}")
    tests = {name: function for name, function in globals().items() if name.startswith("test_") and callable(function)}
    failed = []
    for name, function in tests.items():
        try:
            function()
        except Exception:
            failed.append(name)
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            print(f"pass {name}")
    print(f"{len(tests) - len(failed)}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
