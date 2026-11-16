def cfg(noise_pred_uncond, noise_pred_cond, guidance_scale):
    # Same as the slide's D(c) + (w - 1)(D(c) - D(0)); written in diffusers' order so fp16 results match bit for bit.
    return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
