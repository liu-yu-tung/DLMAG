import importlib.util
import inspect
import json
import math
import os
import sys
import traceback
import zlib
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers import CosineDPMSolverMultistepScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg

from text_to_music import sampler
from text_to_music.cfg import cfg
from text_to_music.sampler import (
    TRAJECTORY_KEYS, Guidance, MomentumBuffer, composed_cfg, generate, guidance_active, guided_output,
    output_coefficients, rescale_to_conditional,
)

HW3 = Path(__file__).resolve().parents[1]
PROJECT = Path(os.environ.get("HW3_PROJECT", HW3))
# Frozen copy of the sampler at commit 6a97bcc (before the guidance extensions); the byte-for-byte checks compare against it.
ORIGINAL_SAMPLER = HW3 / "tests/fixtures/sampler_before_guidance_extensions.py"
STEPS = 12
SAMPLE_SIZE = 64
SECONDS = 0.5


def load_original_sampler():
    spec = importlib.util.spec_from_file_location("sampler_original", ORIGINAL_SAMPLER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_generate_script():
    spec = importlib.util.spec_from_file_location("generate_script", HW3 / "scripts/generation/generate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StubTransformer:
    config = SimpleNamespace(in_channels=4, sample_size=SAMPLE_SIZE)

    def __init__(self):
        self.batch_sizes = []

    def __call__(self, model_input, timestep, encoder_hidden_states, global_hidden_states, rotary_embedding, return_dict):
        batch = model_input.shape[0]
        assert batch == encoder_hidden_states.shape[0] == global_hidden_states.shape[0]
        self.batch_sizes.append(batch)
        positions = torch.linspace(0.5, 1.5, model_input.shape[-1])
        prompt_signal = encoder_hidden_states.mean(dim=(1, 2)).view(-1, 1, 1)
        duration_signal = global_hidden_states.mean(dim=(1, 2)).view(-1, 1, 1)
        output = 0.4 * torch.tanh(model_input * positions) + prompt_signal * (1 + 0.1 * duration_signal) - 0.01 * timestep
        return (output,)


class StubPipe:
    rotary_embed_dim = 8

    def __init__(self):
        self.scheduler = CosineDPMSolverMultistepScheduler(
            sigma_min=0.3, sigma_max=500.0, sigma_data=1.0, sigma_schedule="exponential",
            prediction_type="v_prediction", solver_order=2,
        )
        self.transformer = StubTransformer()
        self.vae = SimpleNamespace(
            config=SimpleNamespace(sampling_rate=1000, audio_channels=2, downsampling_ratios=[2, 5]),
            decode=lambda latents: SimpleNamespace(sample=latents.repeat_interleave(10, dim=-1)[:, :2]),
        )
        self._execution_device = torch.device("cpu")

    def encode_prompt(self, prompt, device, do_classifier_free_guidance):
        generator = torch.Generator().manual_seed(zlib.crc32(prompt.encode()))
        return torch.randn(1, 3, 8, generator=generator)

    def encode_duration(self, start, end, device, do_classifier_free_guidance, batch_size):
        return torch.full((1, 1, 8), 0.1 * start), torch.full((1, 1, 8), 0.1 * end)

    def prepare_latents(self, batch, channels, sample_size, dtype, device, generator, audio_channels):
        return torch.randn(batch, channels, sample_size, generator=generator, device=device, dtype=dtype) * self.scheduler.init_noise_sigma

    def prepare_extra_step_kwargs(self, generator, eta):
        accepted = set(inspect.signature(self.scheduler.step).parameters)
        return {"generator": generator} if "generator" in accepted else {}


def run(guidance, generate_function=generate, prompt="a slow piano melody", **kwargs):
    pipe = StubPipe()
    audio, info = generate_function(pipe, prompt, guidance, steps=STEPS, seconds=SECONDS, seed=3, **kwargs)
    return audio, info, pipe


def sigmas_of_run() -> list[float]:
    scheduler = StubPipe().scheduler
    scheduler.set_timesteps(STEPS, device="cpu")
    return [float(sigma) for sigma in scheduler.sigmas[:-1]]


def random_tensors(count: int, dtype=torch.float64, seed: int = 0, shape=(3, 4, 16)) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(shape, generator=generator, dtype=dtype) for _ in range(count)]


def relative_error(value: torch.Tensor, reference: torch.Tensor) -> float:
    return float((value - reference).norm() / reference.norm())


def test_rescale_phi_zero_is_identity():
    guided, conditional = random_tensors(2)
    assert torch.equal(rescale_to_conditional(guided, conditional, 0.0), guided)


def test_rescale_phi_one_matches_conditional_std():
    guided, conditional = random_tensors(2, seed=1)
    guided = guided * 5
    output = rescale_to_conditional(guided, conditional, 1.0)
    dims = [1, 2]
    assert torch.allclose(output.std(dim=dims), conditional.std(dim=dims), rtol=1e-12)
    assert not torch.allclose(guided.std(dim=dims), conditional.std(dim=dims), rtol=0.1)


def test_rescale_matches_formula_and_diffusers():
    guided, conditional = random_tensors(2, seed=2)
    for phi in (0.3, 0.7, 1.0):
        std_ratio = conditional.std(dim=[1, 2], keepdim=True) / guided.std(dim=[1, 2], keepdim=True)
        expected = phi * guided * std_ratio + (1 - phi) * guided
        assert relative_error(rescale_to_conditional(guided, conditional, phi), expected) < 1e-12, phi
        assert relative_error(rescale_to_conditional(guided, conditional, phi), rescale_noise_cfg(guided, conditional, phi)) < 1e-12, phi


def test_interval_predicate_bounds():
    guidance = Guidance(interval_sigma_low=0.5, interval_sigma_high=10.0)
    assert not guidance_active(0.5, guidance)
    assert guidance_active(math.nextafter(0.5, 1.0), guidance)
    assert guidance_active(10.0, guidance)
    assert not guidance_active(math.nextafter(10.0, 11.0), guidance)


def test_default_interval_keeps_every_step_on():
    assert all(guidance_active(sigma, Guidance()) for sigma in sigmas_of_run())
    assert guidance_active(500.0, Guidance()) and guidance_active(1e-6, Guidance())


def test_composed_without_second_share_equals_cfg():
    uncond, cond, second = random_tensors(3, seed=3)
    for scale in (1.0, 3.0, 7.0):
        assert torch.equal(composed_cfg(uncond, cond, second, scale, 0.0), cfg(uncond, cond, scale))


def test_composed_equal_split_is_cfg_toward_the_mean():
    uncond, cond, second = random_tensors(3, seed=4)
    for scale in (2.0, 7.0):
        composed = composed_cfg(uncond, cond, second, scale, scale / 2)
        assert relative_error(composed, cfg(uncond, (cond + second) / 2, scale)) < 1e-12, scale


def test_composed_matches_liu_formula():
    uncond, cond, second = random_tensors(3, seed=5)
    expected = uncond + 4.0 * (cond - uncond) + 3.0 * (second - uncond)
    assert relative_error(composed_cfg(uncond, cond, second, 7.0, 3.0), expected) < 1e-12


def test_guidance_extension_validation():
    bad_settings = (
        {"interval_sigma_low": -1.0},
        {"interval_sigma_low": 5.0, "interval_sigma_high": 5.0},
        {"rescale_phi": 1.5},
        {"rescale_phi": 0.7, "mode": "apg"},
        {"second_scale": -1.0},
        {"second_scale": 2.0, "mode": "apg"},
    )
    for settings in bad_settings:
        try:
            Guidance(**settings)
        except AssertionError:
            continue
        raise AssertionError(f"Guidance accepted {settings}")
    Guidance("cfg", 7.0, interval_sigma_low=0.3, interval_sigma_high=50.0, rescale_phi=0.7, second_scale=3.5)
    Guidance("apg", 7.0, interval_sigma_low=0.3, interval_sigma_high=50.0)


def test_guided_output_matches_original_loop_body():
    original = load_original_sampler()
    generator = torch.Generator().manual_seed(6)
    shape = (1, 64, 1024)
    sample, v_uncond, v_cond = (torch.randn(shape, generator=generator) for _ in range(3))
    for mode, extras in (("cfg", {}), ("apg", {}), ("apg", {"apg_eta": 0.3, "apg_momentum": -0.5, "apg_norm_threshold": 40.0})):
        for scale in (1.0, 7.0):
            guidance = Guidance(mode, scale, **extras)
            reference_guidance = original.Guidance(mode, scale, **extras)
            for sigma in (0.3, 4.0, 500.0):
                c_skip, c_out = output_coefficients(SimpleNamespace(config=SimpleNamespace(sigma_data=1.0, prediction_type="v_prediction")), torch.tensor(sigma))
                denoised_cond = c_skip * sample + c_out * v_cond.float()
                denoised_uncond = c_skip * sample + c_out * v_uncond.float()
                expected_stats = original.guidance_stats(denoised_cond, denoised_uncond, 215)
                if mode == "cfg":
                    expected = cfg(v_uncond, v_cond, scale)
                else:
                    denoised = original.apg(denoised_cond, denoised_uncond, reference_guidance, original.MomentumBuffer(guidance.apg_momentum))
                    expected = ((denoised - c_skip * sample) / c_out).to(v_cond.dtype)
                v, stats = guided_output(v_uncond, v_cond, None, sample, c_skip, c_out, guidance, MomentumBuffer(guidance.apg_momentum), 215)
                assert torch.equal(v, expected), (mode, extras, scale, sigma)
                assert torch.equal(stats, expected_stats)


def test_default_generate_equals_original_for_cfg_and_apg():
    original = load_original_sampler()
    cases = (
        ("cfg", {}),
        ("apg", {"apg_eta": 0.0}),
        ("apg", {"apg_eta": 0.2, "apg_momentum": -0.5, "apg_norm_threshold": 20.0}),
    )
    for mode, extras in cases:
        audio, info, pipe = run(Guidance(mode, 5.0, **extras))
        reference_audio, reference_info, _ = run(original.Guidance(mode, 5.0, **extras), original.generate)
        assert torch.equal(audio, reference_audio), (mode, extras)
        assert info.pop("forward_samples") == 2 * STEPS
        assert info == reference_info, (mode, extras)
        assert pipe.transformer.batch_sizes == [2] * STEPS
        assert all(value is not None for values in info["trajectory"].values() for value in values)


def test_settings_of_default_run_are_unchanged():
    default = Guidance()
    assert list(default.settings()) == ["mode", "scale", "apg_eta", "apg_momentum", "apg_norm_threshold"]
    assert Guidance("apg", 3.0, 0.1, -0.5, 9.0).settings() == {
        "mode": "apg", "scale": 3.0, "apg_eta": 0.1, "apg_momentum": -0.5, "apg_norm_threshold": 9.0}
    changed = Guidance(interval_sigma_high=50.0, rescale_phi=0.7).settings()
    assert changed["interval_sigma_high"] == 50.0 and changed["rescale_phi"] == 0.7
    assert "interval_sigma_low" not in changed and "second_scale" not in changed
    json.dumps(changed, allow_nan=False)


def test_default_settings_equal_stored_gen_info():
    stored = json.loads((PROJECT / "results/test__cfg7/gen_info.json").read_text())["settings"]
    script = load_generate_script()
    sys.argv = ["generate.py", "--run", "cfg7"]
    args = script.parse_args()
    settings = script.build_settings(args, script.build_guidance(args), script.resolve_compile("off"))
    assert settings == stored, (settings, stored)
    assert json.dumps(settings, sort_keys=True) == json.dumps(stored, sort_keys=True)


def test_new_options_extend_settings_only_when_set():
    script = load_generate_script()
    sys.argv = ["generate.py", "--run", "x", "--rescale-phi", "0.7", "--second-prompt-style", "tags", "--second-scale", "3.5",
                "--interval-sigma-low", "0.5", "--interval-sigma-high", "80"]
    args = script.parse_args()
    settings = script.build_settings(args, script.build_guidance(args), False)
    assert settings["second_prompt_style"] == "tags"
    assert settings["guidance"] == {
        "mode": "cfg", "scale": 7.0, "apg_eta": 0.0, "apg_momentum": 0.0, "apg_norm_threshold": 0.0,
        "interval_sigma_low": 0.5, "interval_sigma_high": 80.0, "rescale_phi": 0.7, "second_scale": 3.5}
    json.dumps(settings, allow_nan=False)
    sys.argv = ["generate.py", "--run", "x", "--second-scale", "3.5"]
    try:
        script.build_guidance(script.parse_args())
    except AssertionError:
        return
    raise AssertionError("--second-scale without --second-prompt-style was accepted")


def test_interval_saves_forward_samples_and_nulls_the_stats():
    sigmas = sigmas_of_run()
    high = sigmas[STEPS // 2]
    low = sigmas[-3]
    guidance = Guidance(interval_sigma_low=low, interval_sigma_high=high)
    audio, info, pipe = run(guidance)
    on_steps = [guidance_active(sigma, guidance) for sigma in sigmas]
    assert 0 < sum(on_steps) < STEPS
    assert info["forward_samples"] == sum(2 if on else 1 for on in on_steps)
    assert pipe.transformer.batch_sizes == [2 if on else 1 for on in on_steps]
    for key in TRAJECTORY_KEYS:
        assert [value is not None for value in info["trajectory"][key]] == on_steps, key
    json.dumps(info, allow_nan=False)
    assert torch.isfinite(audio).all()


def test_interval_that_never_guides_equals_scale_one():
    never = Guidance(interval_sigma_low=1e6)
    audio, info, pipe = run(never)
    assert info["forward_samples"] == STEPS and pipe.transformer.batch_sizes == [1] * STEPS
    assert all(value is None for values in info["trajectory"].values() for value in values)
    reference_audio, _, _ = run(Guidance(scale=1.0))
    assert relative_error(audio, reference_audio) < 1e-4


def test_interval_that_always_guides_equals_default():
    wide = Guidance(interval_sigma_low=0.01, interval_sigma_high=1e6)
    audio, info, _ = run(wide)
    reference_audio, reference_info, _ = run(Guidance())
    assert torch.equal(audio, reference_audio)
    assert info == reference_info


def test_interval_works_with_apg():
    sigmas = sigmas_of_run()
    guidance = Guidance("apg", 5.0, apg_momentum=-0.5, interval_sigma_high=sigmas[STEPS // 2])
    audio, info, _ = run(guidance)
    assert info["forward_samples"] < 2 * STEPS
    assert torch.isfinite(audio).all()
    json.dumps(info, allow_nan=False)


def test_rescale_in_the_loop():
    default_audio, _, _ = run(Guidance(scale=7.0))
    off_audio, _, _ = run(Guidance(scale=7.0, rescale_phi=0.0))
    assert torch.equal(off_audio, default_audio)
    rescaled_audio, info, _ = run(Guidance(scale=7.0, rescale_phi=0.7))
    assert info["forward_samples"] == 2 * STEPS
    assert torch.isfinite(rescaled_audio).all() and not torch.equal(rescaled_audio, default_audio)


def test_composed_in_the_loop():
    guidance = Guidance(scale=6.0, second_scale=3.0)
    audio, info, pipe = run(guidance, second_prompt="piano, slow, calm")
    assert info["forward_samples"] == 3 * STEPS and pipe.transformer.batch_sizes == [3] * STEPS
    default_audio, default_info, _ = run(Guidance(scale=6.0))
    assert not torch.equal(audio, default_audio)
    assert info["trajectory"] == default_info["trajectory"] or all(
        len(values) == STEPS for values in info["trajectory"].values())
    same_prompt_audio, _, _ = run(guidance, second_prompt="a slow piano melody")
    assert relative_error(same_prompt_audio, default_audio) < 1e-4
    for bad in ({"guidance": Guidance(scale=6.0, second_scale=3.0)}, {"guidance": Guidance(scale=6.0), "second_prompt": "x"}):
        try:
            run(bad.pop("guidance"), **bad)
        except AssertionError:
            continue
        raise AssertionError("second prompt and second scale were allowed apart")


def test_composed_with_interval_runs_conditional_only_outside():
    sigmas = sigmas_of_run()
    guidance = Guidance(scale=6.0, second_scale=3.0, interval_sigma_high=sigmas[STEPS // 2])
    _, info, pipe = run(guidance, second_prompt="piano, slow, calm")
    assert set(pipe.transformer.batch_sizes) == {1, 3}
    assert info["forward_samples"] == sum(pipe.transformer.batch_sizes)


def test_rescale_valid_frames_matches_conditional_std_on_the_clip():
    torch.manual_seed(0)
    guided, conditional = 3 * torch.randn(2, 64, 1024, dtype=torch.float64), torch.randn(2, 64, 1024, dtype=torch.float64)
    # The padded tail carries no guidance difference.
    guided[..., 216:] = conditional[..., 216:]
    dims = (1, 2)
    output = rescale_to_conditional(guided, conditional, 1.0, valid_frames=216)
    assert relative_error(output[..., :216].std(dim=dims), conditional[..., :216].std(dim=dims)) < 1e-12
    whole = rescale_to_conditional(guided, conditional, 1.0)
    assert (whole[..., :216].std(dim=dims) > 1.5 * conditional[..., :216].std(dim=dims)).all()
    assert torch.equal(rescale_to_conditional(guided, conditional, 0.7, valid_frames=None), rescale_to_conditional(guided, conditional, 0.7))


def test_rescale_valid_only_is_an_extension_field():
    assert "rescale_valid_only" not in Guidance(scale=12.0, rescale_phi=0.7).settings()
    assert Guidance(scale=12.0, rescale_phi=0.7, rescale_valid_only=True).settings()["rescale_valid_only"] is True



def test_extra_term_keeps_the_caption_weight():
    uncond, cond, extra = (torch.randn(1, 64, 216) for _ in range(3))
    expected = uncond + 12.0 * (cond - uncond) - 3.0 * (extra - uncond)
    assert torch.allclose(sampler.extra_term_cfg(uncond, cond, extra, 12.0, -3.0), expected, atol=1e-5)
    assert torch.equal(sampler.extra_term_cfg(uncond, cond, extra, 12.0, 0.0), cfg(uncond, cond, 12.0))


def test_extra_term_validation_and_settings():
    assert "extra_scale" not in Guidance(scale=12.0).settings()
    assert Guidance(scale=12.0, extra_scale=-3.0).settings()["extra_scale"] == -3.0
    for bad in (dict(second_scale=2.0, extra_scale=-3.0), dict(mode="apg", extra_scale=-3.0)):
        try:
            Guidance(scale=12.0, **bad)
        except AssertionError:
            continue
        raise AssertionError(f"accepted {bad}")

if __name__ == "__main__":
    torch.manual_seed(0)
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
