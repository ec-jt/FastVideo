# SPDX-License-Identifier: Apache-2.0
"""
LTX-2.3 HQ two-stage denoising with Res2s second-order sampler.

Upstream equivalent: ``TI2VidTwoStagesHQPipeline``

Same two-stage structure as ``LTX2TwoStageDenoisingStage`` but uses
the Res2s second-order sampler instead of Euler, allowing fewer steps
(15 vs 30) for comparable quality.  Both stages use a distilled LoRA
at configurable per-stage strengths.
"""

from __future__ import annotations

from functools import partial

import torch
from tqdm.auto import tqdm

import fastvideo.envs as envs
from fastvideo.attention.backends.video_sparse_attn import (
    VideoSparseAttentionMetadataBuilder)
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.dits.ltx2 import (
    AudioLatentShape, DEFAULT_LTX2_AUDIO_CHANNELS,
    DEFAULT_LTX2_AUDIO_DOWNSAMPLE, DEFAULT_LTX2_AUDIO_HOP_LENGTH,
    DEFAULT_LTX2_AUDIO_MEL_BINS, DEFAULT_LTX2_AUDIO_SAMPLE_RATE,
    VideoLatentShape)
from fastvideo.models.schedulers.res2s import (
    get_res2s_coefficients, res2s_step)
from fastvideo.models.upsamplers.latent_upsampler import (
    LatentUpsampler)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.ltx2_denoising import (
    _ltx2_sigmas)
from fastvideo.pipelines.stages.ltx2_distilled_denoising import (
    STAGE_2_DISTILLED_SIGMA_VALUES)
from fastvideo.pipelines.stages.ltx2_stage_utils import (
    debug_save_latent, noise_latent_at_sigma,
    run_spatial_upsample)
from fastvideo.pipelines.stages.ltx2_two_stage_denoising import (
    LTX2TwoStageDenoisingStage)
from fastvideo.pipelines.stages.validators import (
    StageValidators as V, VerificationResult)
from fastvideo.utils import is_vsa_available

logger = init_logger(__name__)

try:
    vsa_available = is_vsa_available()
except ImportError:
    vsa_available = False


def _channelwise_normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize per-channel to zero mean, unit variance."""
    return x.sub_(
        x.mean(dim=(-2, -1), keepdim=True)
    ).div_(
        x.std(dim=(-2, -1), keepdim=True)
    )


def _get_new_noise(
    x: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Generate normalized noise for SDE injection."""
    noise = torch.randn(
        x.shape,
        generator=generator,
        dtype=torch.float64,
        device=generator.device,
    )
    noise = (noise - noise.mean()) / noise.std()
    return _channelwise_normalize(noise)


class LTX2HQDenoisingStage(LTX2TwoStageDenoisingStage):
    """HQ two-stage denoising with Res2s sampler.

    Inherits from ``LTX2TwoStageDenoisingStage`` and overrides
    Stage 1 to use the Res2s second-order sampler with SDE noise
    injection.  Stage 2 also uses Res2s (matching upstream HQ).
    """

    # ─────────────────────────────────────────────────────────
    # Res2s denoising loop (for both stages in HQ pipeline)
    # ─────────────────────────────────────────────────────────

    def _res2s_guided_denoise_loop(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        neg_prompt_embeds: torch.Tensor | None,
        sigmas: torch.Tensor,
        audio_latents: torch.Tensor | None,
        audio_context_p: torch.Tensor | None,
        audio_context_n: torch.Tensor | None,
        audio_timestep_template: torch.Tensor | None,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
        fastvideo_args: FastVideoArgs,
        batch: ForwardBatch,
        noise_seed: int = 42,
        stage_label: str = "",
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run Res2s second-order denoising with guidance.

        This implements the ``res2s_audio_video_denoising_loop``
        from upstream, adapted to FastVideo's tensor-based approach.
        """
        device = latents.device
        model_dtype = target_dtype

        # Initialize noise generators
        step_gen = torch.Generator(device=device).manual_seed(
            noise_seed)
        substep_gen = torch.Generator(
            device=device).manual_seed(noise_seed + 10000)

        n_full_steps = len(sigmas) - 1
        # Inject minimal sigma to avoid division by zero
        if sigmas[-1] == 0:
            sigmas = torch.cat([
                sigmas[:-1],
                torch.tensor(
                    [0.0011, 0.0], device=device),
            ], dim=0)

        hs = -torch.log(
            sigmas[1:].double().cpu()
            / sigmas[:-1].double().cpu()
        )

        phi_cache: dict = {}
        c2 = 0.5

        for step_idx in tqdm(
            range(n_full_steps),
            desc=f"Res2s {stage_label}",
        ):
            sigma = sigmas[step_idx].double()
            sigma_next = sigmas[step_idx + 1].double()

            x_anchor_v = latents.clone().double()
            x_anchor_a = (
                audio_latents.clone().double()
                if audio_latents is not None else None
            )

            # Stage 1: Evaluate at current point
            d_v1, d_a1 = self._guided_denoise_step(
                latents=latents,
                prompt_embeds=prompt_embeds,
                neg_prompt_embeds=neg_prompt_embeds,
                sigma=sigmas[step_idx],
                audio_latents=audio_latents,
                audio_context_p=audio_context_p,
                audio_context_n=audio_context_n,
                audio_timestep_template=(
                    audio_timestep_template),
                target_dtype=target_dtype,
                autocast_enabled=autocast_enabled,
                fastvideo_args=fastvideo_args,
                batch=batch,
                step_index=step_idx,
            )

            h = hs[step_idx].item()
            a21, b1, b2 = get_res2s_coefficients(
                h, phi_cache, c2)

            sub_sigma = torch.sqrt(sigma * sigma_next)

            # Compute substep x using RK coefficient a21
            eps_1_v = d_v1.double() - x_anchor_v
            x_mid_v = x_anchor_v + h * a21 * eps_1_v

            eps_1_a = None
            x_mid_a = None
            if x_anchor_a is not None and d_a1 is not None:
                eps_1_a = d_a1.double() - x_anchor_a
                x_mid_a = x_anchor_a + h * a21 * eps_1_a

            # SDE noise injection at substep
            sub_sigmas = torch.stack([sigma, sub_sigma])
            noise_v = _get_new_noise(latents, substep_gen)
            x_mid_v = res2s_step(
                sample=x_anchor_v,
                denoised_sample=x_mid_v,
                sigma=sub_sigmas[0],
                sigma_next=sub_sigmas[1],
                noise=noise_v,
            )

            if x_mid_a is not None:
                noise_a = _get_new_noise(
                    audio_latents, substep_gen)
                x_mid_a = res2s_step(
                    sample=x_anchor_a,
                    denoised_sample=x_mid_a,
                    sigma=sub_sigmas[0],
                    sigma_next=sub_sigmas[1],
                    noise=noise_a,
                )

            # Bong iteration (anchor refinement)
            if h < 0.5 and sigma > 0.03:
                for _ in range(100):
                    x_anchor_v = (
                        x_mid_v - h * a21 * eps_1_v)
                    eps_1_v = d_v1.double() - x_anchor_v
                    if x_mid_a is not None:
                        x_anchor_a = (
                            x_mid_a - h * a21 * eps_1_a)
                        eps_1_a = d_a1.double() - x_anchor_a

            # Stage 2: Evaluate at substep point
            mid_latents = x_mid_v.to(model_dtype)
            mid_audio = (
                x_mid_a.to(model_dtype)
                if x_mid_a is not None else None
            )

            d_v2, d_a2 = self._guided_denoise_step(
                latents=mid_latents,
                prompt_embeds=prompt_embeds,
                neg_prompt_embeds=neg_prompt_embeds,
                sigma=sub_sigma.to(
                    device=device, dtype=torch.float32),
                audio_latents=mid_audio,
                audio_context_p=audio_context_p,
                audio_context_n=audio_context_n,
                audio_timestep_template=(
                    audio_timestep_template),
                target_dtype=target_dtype,
                autocast_enabled=autocast_enabled,
                fastvideo_args=fastvideo_args,
                batch=batch,
                step_index=step_idx,
            )

            # Final combination using RK coefficients
            eps_2_v = d_v2.double() - x_anchor_v
            x_next_v = (
                x_anchor_v
                + h * (b1 * eps_1_v + b2 * eps_2_v)
            )

            x_next_a = None
            if x_anchor_a is not None and d_a2 is not None:
                eps_2_a = d_a2.double() - x_anchor_a
                x_next_a = (
                    x_anchor_a
                    + h * (b1 * eps_1_a + b2 * eps_2_a)
                )

            # SDE noise injection at step level
            noise_v = _get_new_noise(latents, step_gen)
            x_next_v = res2s_step(
                sample=x_anchor_v,
                denoised_sample=x_next_v,
                sigma=sigmas[step_idx],
                sigma_next=sigmas[step_idx + 1],
                noise=noise_v,
            )

            if x_next_a is not None:
                noise_a = _get_new_noise(
                    audio_latents, step_gen)
                x_next_a = res2s_step(
                    sample=x_anchor_a,
                    denoised_sample=x_next_a,
                    sigma=sigmas[step_idx],
                    sigma_next=sigmas[step_idx + 1],
                    noise=noise_a,
                )

            latents = x_next_v.to(model_dtype)
            audio_latents = (
                x_next_a.to(model_dtype)
                if x_next_a is not None else audio_latents
            )

        # Final step if sigma[-1] == 0
        if sigmas[-1] == 0:
            d_v, d_a = self._guided_denoise_step(
                latents=latents,
                prompt_embeds=prompt_embeds,
                neg_prompt_embeds=neg_prompt_embeds,
                sigma=sigmas[n_full_steps],
                audio_latents=audio_latents,
                audio_context_p=audio_context_p,
                audio_context_n=audio_context_n,
                audio_timestep_template=(
                    audio_timestep_template),
                target_dtype=target_dtype,
                autocast_enabled=autocast_enabled,
                fastvideo_args=fastvideo_args,
                batch=batch,
                step_index=n_full_steps,
            )
            latents = d_v.to(model_dtype)
            if d_a is not None:
                audio_latents = d_a.to(model_dtype)

        return latents, audio_latents

    def _guided_denoise_step(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        neg_prompt_embeds: torch.Tensor | None,
        sigma: torch.Tensor,
        audio_latents: torch.Tensor | None,
        audio_context_p: torch.Tensor | None,
        audio_context_n: torch.Tensor | None,
        audio_timestep_template: torch.Tensor | None,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
        fastvideo_args: FastVideoArgs,
        batch: ForwardBatch,
        step_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Single guided denoising evaluation (no Euler step).

        Returns ``(denoised_video, denoised_audio)`` after
        applying CFG/STG/modality guidance.
        """
        if hasattr(self.transformer, "patchifier"):
            video_shape = VideoLatentShape.from_torch_shape(
                latents.shape)
            token_count = (
                self.transformer.patchifier.get_token_count(
                    video_shape))
        else:
            token_count = 1

        timestep_template = torch.ones(
            (latents.shape[0], token_count),
            device=latents.device,
            dtype=torch.float32,
        )

        if isinstance(sigma, torch.Tensor):
            sigma_val = sigma.to(torch.float32)
        else:
            sigma_val = torch.tensor(
                float(sigma),
                device=latents.device,
                dtype=torch.float32,
            )

        timestep = timestep_template * sigma_val
        audio_ts = (
            audio_timestep_template * sigma_val
            if audio_timestep_template is not None
            else None
        )

        cfg_v = batch.ltx2_cfg_scale_video
        cfg_a = batch.ltx2_cfg_scale_audio
        mod_v = batch.ltx2_modality_scale_video
        mod_a = batch.ltx2_modality_scale_audio
        rescale = batch.ltx2_rescale_scale
        stg_v = batch.ltx2_stg_scale_video
        stg_a = batch.ltx2_stg_scale_audio
        stg_blk_v = batch.ltx2_stg_blocks_video
        stg_blk_a = batch.ltx2_stg_blocks_audio
        do_stg_v = stg_v != 0.0
        do_stg_a = stg_a != 0.0
        do_stg = do_stg_v or do_stg_a
        do_cfg = cfg_v != 1.0 or cfg_a != 1.0
        do_mod = mod_v != 1.0 or mod_a != 1.0
        do_guidance = do_cfg or do_mod or do_stg

        with torch.autocast(
            device_type="cuda",
            dtype=target_dtype,
            enabled=autocast_enabled,
        ), set_forward_context(
            current_timestep=sigma_val.item(),
            attn_metadata=None,
            forward_batch=batch,
        ):
            pos_out = self.transformer(
                hidden_states=latents.to(target_dtype),
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=None,
                timestep=timestep,
                audio_hidden_states=audio_latents,
                audio_encoder_hidden_states=audio_context_p,
                audio_timestep=audio_ts,
            )
            if isinstance(pos_out, tuple):
                pos_vid, pos_aud = pos_out
            else:
                pos_vid = pos_out
                pos_aud = None

            if do_guidance:
                neg_vid = pos_vid
                neg_aud = pos_aud
                mod_vid_d = pos_vid
                mod_aud_d = pos_aud
                ptb_vid = pos_vid
                ptb_aud = pos_aud

                if do_cfg and neg_prompt_embeds is not None:
                    neg_out = self.transformer(
                        hidden_states=latents.to(
                            target_dtype),
                        encoder_hidden_states=(
                            neg_prompt_embeds),
                        encoder_attention_mask=None,
                        timestep=timestep,
                        audio_hidden_states=audio_latents,
                        audio_encoder_hidden_states=(
                            audio_context_n),
                        audio_timestep=audio_ts,
                    )
                    if isinstance(neg_out, tuple):
                        neg_vid, neg_aud = neg_out
                    else:
                        neg_vid = neg_out

                if do_mod:
                    mod_out = self.transformer(
                        hidden_states=latents.to(
                            target_dtype),
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=None,
                        timestep=timestep,
                        audio_hidden_states=audio_latents,
                        audio_encoder_hidden_states=(
                            audio_context_p),
                        audio_timestep=audio_ts,
                        skip_cross_modal_attn=True,
                    )
                    if isinstance(mod_out, tuple):
                        mod_vid_d, mod_aud_d = mod_out
                    else:
                        mod_vid_d = mod_out

                if do_stg:
                    ptb_out = self.transformer(
                        hidden_states=latents.to(
                            target_dtype),
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=None,
                        timestep=timestep,
                        audio_hidden_states=audio_latents,
                        audio_encoder_hidden_states=(
                            audio_context_p),
                        audio_timestep=audio_ts,
                        skip_video_self_attn_blocks=(
                            stg_blk_v
                            if do_stg_v else None),
                        skip_audio_self_attn_blocks=(
                            stg_blk_a
                            if do_stg_a else None),
                    )
                    if isinstance(ptb_out, tuple):
                        ptb_vid, ptb_aud = ptb_out
                    else:
                        ptb_vid = ptb_out

                vid = (
                    pos_vid
                    + (cfg_v - 1) * (pos_vid - neg_vid)
                    + (mod_v - 1) * (pos_vid - mod_vid_d)
                    + stg_v * (pos_vid - ptb_vid)
                )
                aud = None
                if pos_aud is not None:
                    aud = (
                        pos_aud
                        + (cfg_a - 1)
                        * (pos_aud - neg_aud)
                        + (mod_a - 1)
                        * (pos_aud - mod_aud_d)
                        + stg_a * (pos_aud - ptb_aud)
                    )

                if rescale > 0:
                    f_v = pos_vid.std() / vid.std()
                    f_v = rescale * f_v + (1 - rescale)
                    vid = vid * f_v
                    if aud is not None:
                        f_a = pos_aud.std() / aud.std()
                        f_a = rescale * f_a + (1 - rescale)
                        aud = aud * f_a

                pos_vid = vid
                pos_aud = aud

        return pos_vid, pos_aud

    # ─────────────────────────────────────────────────────────
    # Override two-stage to use Res2s
    # ─────────────────────────────────────────────────────────

    def _run_two_stage(
        self,
        batch: ForwardBatch,
        prompt_embeds: torch.Tensor,
        neg_prompt_embeds: torch.Tensor | None,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
        fastvideo_args: FastVideoArgs,
    ) -> None:
        full_latents = batch.latents
        assert full_latents is not None
        device = full_latents.device

        b, c, f, h, w = full_latents.shape
        half_h, half_w = h // 2, w // 2

        sigmas = _ltx2_sigmas(
            steps=batch.num_inference_steps,
            latent=None,
            device=device,
        )

        logger.info(
            "[LTX2-HQ] Stage 1: half-res %dx%d "
            "(%d steps, Res2s guided)",
            half_h, half_w, len(sigmas) - 1,
        )

        gen = None
        if batch.seed is not None:
            gen = torch.Generator(device=device).manual_seed(
                batch.seed)
        half_latents = torch.randn(
            (b, c, f, half_h, half_w),
            generator=gen,
            device=device,
            dtype=full_latents.dtype,
        )

        (audio_latents, audio_ctx_p,
         audio_ts_tpl) = self._init_audio_latents(
            batch, half_latents, fastvideo_args,
            noise_scale=1.0,
        )
        audio_neg = batch.extra.get(
            "ltx2_audio_negative_embeds")
        audio_ctx_n = audio_neg[0] if audio_neg else None

        orig_h = batch.height
        orig_w = batch.width
        sr = (fastvideo_args.pipeline_config.vae_config
              .arch_config.spatial_compression_ratio)
        batch.height = half_h * sr
        batch.width = half_w * sr

        noise_seed = batch.seed if batch.seed is not None else 42

        half_latents, audio_latents = (
            self._res2s_guided_denoise_loop(
                latents=half_latents,
                prompt_embeds=prompt_embeds,
                neg_prompt_embeds=neg_prompt_embeds,
                sigmas=sigmas,
                audio_latents=audio_latents,
                audio_context_p=audio_ctx_p,
                audio_context_n=audio_ctx_n,
                audio_timestep_template=audio_ts_tpl,
                target_dtype=target_dtype,
                autocast_enabled=autocast_enabled,
                fastvideo_args=fastvideo_args,
                batch=batch,
                noise_seed=noise_seed,
                stage_label="Stage 1 (half-res)",
            ))

        batch.height = orig_h
        batch.width = orig_w

        debug_save_latent(
            half_latents, "hq_s1", audio_latents)

        # Spatial upsample
        logger.info(
            "[LTX2-HQ] Upsampling 2x (%dx%d -> %dx%d)",
            half_h, half_w, h, w,
        )
        assert self.spatial_upsampler is not None
        assert self.per_channel_statistics is not None

        upsampled = run_spatial_upsample(
            half_latents=half_latents,
            spatial_upsampler=self.spatial_upsampler,
            per_channel_statistics=(
                self.per_channel_statistics),
            target_h=h,
            target_w=w,
            batch_size=b,
        )

        # Stage 2: full-res with distilled LoRA (Res2s)
        s2_sigmas = torch.tensor(
            STAGE_2_DISTILLED_SIGMA_VALUES,
            device=device,
            dtype=torch.float32,
        )

        logger.info(
            "[LTX2-HQ] Stage 2: full-res %dx%d "
            "(%d steps, Res2s simple)",
            h, w, len(STAGE_2_DISTILLED_SIGMA_VALUES) - 1,
        )

        start_sigma = s2_sigmas[0]
        s2_latents = noise_latent_at_sigma(
            upsampled, start_sigma)

        if audio_latents is not None:
            audio_latents = noise_latent_at_sigma(
                audio_latents, start_sigma)

        if audio_latents is not None:
            audio_ts_tpl = torch.ones(
                (b, audio_latents.shape[2]
                 if audio_latents.dim() == 4
                 else audio_latents.shape[1]),
                device=device,
                dtype=torch.float32,
            )

        # Stage 2 uses simple denoising (no guidance)
        s2_latents, audio_latents = (
            self._simple_denoise_loop(
                latents=s2_latents,
                prompt_embeds=prompt_embeds,
                sigmas=s2_sigmas,
                audio_latents=audio_latents,
                audio_context=audio_ctx_p,
                audio_timestep_template=audio_ts_tpl,
                target_dtype=target_dtype,
                autocast_enabled=autocast_enabled,
                fastvideo_args=fastvideo_args,
                batch=batch,
                stage_label="Stage 2 (full-res, simple)",
            ))

        debug_save_latent(
            s2_latents, "hq_s2", audio_latents)

        batch.latents = s2_latents
        batch.extra["ltx2_audio_latents"] = audio_latents
