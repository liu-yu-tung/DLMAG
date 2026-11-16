"""Stable Audio Open sampling loop with the CFG step taken out of the pipeline.

This mirrors `diffusers.StableAudioPipeline.__call__` (diffusers==0.41.0); the only difference is that
the two model outputs of each step are combined by `cfg` from cfg.py.

    python generate.py --prompt "jazz, electric guitar, piano, relaxing" --out out.wav
    python generate.py --prompt "jazz, electric guitar, piano, relaxing" --check   # compare with diffusers
"""

import argparse

import soundfile as sf
import torch
from diffusers import StableAudioPipeline
from diffusers.models.embeddings import get_1d_rotary_pos_embed

from cfg import cfg


@torch.no_grad()
def generate(pipe, prompt, guidance_scale=7.0, num_inference_steps=100, audio_end_in_s=10.0, seed=0):
    device = pipe._execution_device
    sampling_rate = pipe.vae.config.sampling_rate

    # 1. Conditions: [uncond, cond]. The unconditional branch is an all-zero text+duration embedding.
    prompt_embeds = pipe.encode_prompt(prompt, device, do_classifier_free_guidance=False)
    seconds_start, seconds_end = pipe.encode_duration(0.0, audio_end_in_s, device, False, batch_size=1)
    cond = torch.cat([prompt_embeds, seconds_start, seconds_end], dim=1)
    encoder_hidden_states = torch.cat([torch.zeros_like(cond), cond])
    global_hidden_states = torch.cat([seconds_start, seconds_end], dim=2).repeat(2, 1, 1)

    # 2. Timesteps, initial noise, rotary embedding
    generator = torch.Generator(device).manual_seed(seed)
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    latents = pipe.prepare_latents(
        1, pipe.transformer.config.in_channels, int(pipe.transformer.config.sample_size),
        cond.dtype, device, generator, audio_channels=pipe.vae.config.audio_channels,
    )
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, 0.0)
    rotary_embedding = get_1d_rotary_pos_embed(
        pipe.rotary_embed_dim, latents.shape[2] + global_hidden_states.shape[1],
        use_real=True, repeat_interleave_real=False,
    )

    # 3. Denoising loop
    for t in pipe.progress_bar(pipe.scheduler.timesteps):
        latent_model_input = pipe.scheduler.scale_model_input(torch.cat([latents] * 2), t)
        noise_pred = pipe.transformer(
            latent_model_input, t.unsqueeze(0),
            encoder_hidden_states=encoder_hidden_states,
            global_hidden_states=global_hidden_states,
            rotary_embedding=rotary_embedding,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = cfg(noise_pred_uncond, noise_pred_cond, guidance_scale)  # <- your implementation
        latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

    # 4. Decode
    audio = pipe.vae.decode(latents).sample
    return audio[0, :, : int(audio_end_in_s * sampling_rate)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--scale", type=float, default=7.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="out.wav")
    parser.add_argument("--check", action="store_true", help="compare with StableAudioPipeline")
    args = parser.parse_args()

    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    audio = generate(pipe, args.prompt, args.scale, args.steps, args.seconds, args.seed)
    sf.write(args.out, audio.T.float().cpu().numpy(), pipe.vae.config.sampling_rate)
    print(f"Saved to {args.out}")

    if args.check:
        ref = pipe(args.prompt, num_inference_steps=args.steps, audio_end_in_s=args.seconds,
                   guidance_scale=args.scale, generator=torch.Generator("cuda").manual_seed(args.seed)).audios[0]
        print(f"max |diffusers - ours| = {(ref.float() - audio.float()).abs().max().item():.3e} (0 = identical)")


if __name__ == "__main__":
    main()
