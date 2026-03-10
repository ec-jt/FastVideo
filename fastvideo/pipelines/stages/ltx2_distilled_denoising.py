 # SPDX-License-Identifier: Apache-2.0
"""
LTX-2 distilled two-stage denoising stage.

The distilled pipeline is a two-stage architecture:
  Stage 1 – Generate at **half** spatial resolution with 8 steps.
  Upsample – 2× spatial upsampling in latent space.
  Stage 2 – Refine at **full** resolution with 3 steps.

This matches the official ``DistilledPipeline`` from:
  LTX-2/packages/ltx-pipelines/src/ltx_pipelines/distilled.py
"""

from __future__ import annotations

import os
import threading

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
from fastvideo.models.upsamplers.latent_upsampler import (
    LatentUpsampler, upsample_video_latent)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import (
    StageValidators as V)
from fastvideo.pipelines.stages.validators import (
    VerificationResult)
from fastvideo.utils import is_vsa_available

logger = init_logger(__name__)


def _log_gpu_memory(label: str) -> None:
    """Log current GPU memory usage at a pipeline transition."""
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    logger.info(
        "[GPU-Mem] %s: allocated=%.2f GB, reserved=%.2f GB",
        label, alloc, reserved,
    )


# Debug output directory (set LTX2_DEBUG_DIR env var to enable)
_DEBUG_DIR = os.environ.get("LTX2_DEBUG_DIR", "")


def _debug_save_latent(
    latent: torch.Tensor,
    label: str,
    audio_latent: torch.Tensor | None = None,
) -> None:
    """Save latent tensor for debugging when LTX2_DEBUG_DIR is set."""
    if not _DEBUG_DIR:
        return
    os.makedirs(_DEBUG_DIR, exist_ok=True)
    path = os.path.join(_DEBUG_DIR, f"{label}_latent.pt")
    save_dict = {"video_latent": latent.detach().cpu()}
    if audio_latent is not None:
        save_dict["audio_latent"] = audio_latent.detach().cpu()
    torch.save(save_dict, path)
    logger.info(
        "[DEBUG] Saved %s latent: shape=%s to %s",
        label, tuple(latent.shape), path,
    )





def _debug_decode_and_save(
    latent: torch.Tensor,
    vae,
    label: str,
    fps: int = 24,
) -> None:
    """Decode latent via VAE and save as MP4.

    All workers must call this together (FSDP requirement).
    Only rank 0 writes the file.
    """
    if not _DEBUG_DIR or vae is None:
        return
    from fastvideo.distributed import get_local_torch_device
    device = get_local_torch_device()
    try:
        vae_on_device = vae.to(device)
        vae_dtype = torch.bfloat16
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=vae_dtype, enabled=True,
        ):
            decoded = vae_on_device.decode(
                latent[:1].to(device=device, dtype=vae_dtype))
        decoded = (decoded.float() / 2 + 0.5).clamp(0, 1)
        decoded = (decoded * 255).to(torch.uint8)
        # [B, C, F, H, W] -> [F, H, W, C]
        video = decoded[0].permute(1, 2, 3, 0).cpu().numpy()
        # Only rank 0 saves
        rank = torch.distributed.get_rank() if (
            torch.distributed.is_initialized()) else 0
        if rank == 0:
            os.makedirs(_DEBUG_DIR, exist_ok=True)
            out_path = os.path.join(
                _DEBUG_DIR, f"{label}.mp4")
            import imageio
            writer = imageio.get_writer(
                out_path, fps=fps, codec="libx264",
                quality=8)
            for frame in video:
                writer.append_data(frame)
            writer.close()
            logger.info(
                "[DEBUG] Saved %s video: %d frames "
                "%dx%d to %s",
                label, video.shape[0], video.shape[2],
                video.shape[1], out_path,
            )
    except Exception as e:
        logger.warning(
            "[DEBUG] Failed to save %s video: %s",
            label, e)


# Official distilled sigma schedule (8 denoising steps).
# From LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py
DISTILLED_SIGMA_VALUES = [
    1.0, 0.99375, 0.9875, 0.98125, 0.975,
    0.909375, 0.725, 0.421875, 0.0,
]

# Reduced schedule for super-resolution stage 2 (3 refinement steps).
STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]

try:
    vsa_available = is_vsa_available()
except ImportError:
    vsa_available = False


class LTX2DistilledDenoisingStage(PipelineStage):
    """Run the LTX-2 distilled two-stage denoising loop.

    Stage 1: Denoise at half spatial resolution (8 steps).
    Upsample: 2× spatial upsampling via ``LatentUpsampler``.
    Stage 2: Refine at full resolution (3 steps).

    Uses fixed sigma schedules and simple denoising (single forward
    pass per step, no CFG/STG/modality guidance).
    """

    def __init__(
        self,
        transformer,
        spatial_upsampler: LatentUpsampler | None = None,
        per_channel_statistics=None,
        vae=None,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.spatial_upsampler = spatial_upsampler
        self.per_channel_statistics = per_channel_statistics
        self._debug_vae = vae

    # ─────────────────────────────────────────────────────────
    # Core denoising loop (shared by both stages)
    # ─────────────────────────────────────────────────────────

    def _denoise_loop(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        sigmas: torch.Tensor,
        audio_latents: torch.Tensor | None,
        audio_context: torch.Tensor | None,
        audio_timestep_template: torch.Tensor | None,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
        fastvideo_args: FastVideoArgs,
        batch: ForwardBatch,
        stage_label: str = "",
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the Euler denoising loop over *sigmas*.

        Returns the denoised ``(video_latents, audio_latents)``
        tuple.
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

        use_vsa = (
            vsa_available
            and envs.FASTVIDEO_ATTENTION_BACKEND
            == "VIDEO_SPARSE_ATTN"
        )
        vsa_metadata_builder = (
            VideoSparseAttentionMetadataBuilder()
            if use_vsa
            else None
        )

        for step_index in tqdm(
            range(len(sigmas) - 1),
            desc=f"Denoising {stage_label}",
        ):
            sigma = sigmas[step_index]
            sigma_next = sigmas[step_index + 1]
            timestep = timestep_template * sigma
            audio_timestep = (
                audio_timestep_template * sigma
                if audio_timestep_template is not None
                else None
            )
            attn_metadata = None
            if vsa_metadata_builder is not None:
                attn_metadata = vsa_metadata_builder.build(
                    current_timestep=step_index,
                    raw_latent_shape=latents.shape[2:5],
                    patch_size=(
                        fastvideo_args.pipeline_config
                        .dit_config.patch_size),
                    VSA_sparsity=(
                        fastvideo_args.VSA_sparsity),
                    device=latents.device,
                )

            with torch.autocast(
                device_type="cuda",
                dtype=target_dtype,
                enabled=autocast_enabled,
            ), set_forward_context(
                current_timestep=sigma.item(),
                attn_metadata=attn_metadata,
                forward_batch=batch,
            ):
                outputs = self.transformer(
                    hidden_states=latents.to(target_dtype),
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=None,
                    timestep=timestep,
                    audio_hidden_states=audio_latents,
                    audio_encoder_hidden_states=audio_context,
                    audio_timestep=audio_timestep,
                )
                if isinstance(outputs, tuple):
                    denoised, audio_denoised = outputs
                else:
                    denoised = outputs
                    audio_denoised = None

            # Euler step: velocity = (x - x0) / sigma
            #             x_next  = x + velocity * dt
            sigma_f = sigma.to(torch.float32)
            dt = sigma_next - sigma
            velocity = (
                (latents.float() - denoised.float())
                / sigma_f
            ).to(latents.dtype)
            latents = (
                latents.float() + velocity.float() * dt
            ).to(latents.dtype)

            if (audio_denoised is not None
                    and audio_latents is not None):
                audio_velocity = (
                    (audio_latents.float()
                     - audio_denoised.float())
                    / sigma_f
                ).to(audio_latents.dtype)
                audio_latents = (
                    audio_latents.float()
                    + audio_velocity.float() * dt
                ).to(audio_latents.dtype)

        return latents, audio_latents

    # ─────────────────────────────────────────────────────────
    # Audio latent initialisation
    # ─────────────────────────────────────────────────────────

    def _init_audio_latents(
        self,
        batch: ForwardBatch,
        latents: torch.Tensor,
        fastvideo_args: FastVideoArgs,
        noise_scale: float = 1.0,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Create initial audio latents and timestep template.

        Returns ``(audio_latents, audio_context, timestep_tpl)``.
        """
        audio_prompt_embeds = batch.extra.get(
            "ltx2_audio_prompt_embeds")
        audio_context = (
            audio_prompt_embeds[0]
            if audio_prompt_embeds else None
        )
        if audio_context is None:
            return None, None, None

        fps_value = batch.fps
        if isinstance(fps_value, list):
            fps_value = fps_value[0] if fps_value else None
        if fps_value is None:
            fps_value = 1.0
        duration = float(batch.num_frames) / float(fps_value)
        audio_shape = AudioLatentShape.from_duration(
            batch=latents.shape[0],
            duration=duration,
            channels=DEFAULT_LTX2_AUDIO_CHANNELS,
            mel_bins=DEFAULT_LTX2_AUDIO_MEL_BINS,
            sample_rate=DEFAULT_LTX2_AUDIO_SAMPLE_RATE,
            hop_length=DEFAULT_LTX2_AUDIO_HOP_LENGTH,
            audio_latent_downsample_factor=(
                DEFAULT_LTX2_AUDIO_DOWNSAMPLE),
        )

        # Build generator
        audio_generator = None
        if (fastvideo_args.ltx2_initial_latent_path
                and batch.seed is not None):
            audio_generator = torch.Generator(
                device=latents.device,
            ).manual_seed(batch.seed)
        elif batch.generator is not None:
            if isinstance(batch.generator, list):
                audio_generator = batch.generator[0]
            else:
                audio_generator = batch.generator
        if (audio_generator is not None
                and audio_generator.device.type
                != latents.device.type):
            if batch.seed is None:
                audio_generator = torch.Generator(
                    device=latents.device)
            else:
                audio_generator = torch.Generator(
                    device=latents.device,
                ).manual_seed(batch.seed)

        audio_patch_shape = (
            audio_shape.batch,
            audio_shape.frames,
            audio_shape.channels * audio_shape.mel_bins,
        )
        audio_latents_patch = torch.randn(
            audio_patch_shape,
            generator=audio_generator,
            device=latents.device,
            dtype=latents.dtype,
        ) * noise_scale

        if hasattr(self.transformer, "audio_patchifier"):
            audio_latents = (
                self.transformer.audio_patchifier.unpatchify(
                    audio_latents_patch, audio_shape))
        else:
            audio_latents = audio_latents_patch.view(
                audio_shape.batch,
                audio_shape.frames,
                audio_shape.channels,
                audio_shape.mel_bins,
            ).permute(0, 2, 1, 3).contiguous()

        audio_timestep_template = torch.ones(
            (latents.shape[0], audio_shape.frames),
            device=latents.device,
            dtype=torch.float32,
        )

        return audio_latents, audio_context, audio_timestep_template

    # ─────────────────────────────────────────────────────────
    # Main forward
    # ─────────────────────────────────────────────────────────

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        if batch.latents is None:
            raise ValueError(
                "Latents must be provided before denoising.")

        prompt_embeds = batch.prompt_embeds[0]
        if prompt_embeds.device != batch.latents.device:
            prompt_embeds = prompt_embeds.to(
                batch.latents.device)

        target_dtype = torch.bfloat16
        autocast_enabled = (
            target_dtype != torch.float32
        ) and not fastvideo_args.disable_autocast

        has_upsampler = (
            self.spatial_upsampler is not None
            and self.per_channel_statistics is not None
        )

        # Two-stage is opt-in via env var (experimental).
        # Default is single-stage which produces good results.
        use_two_stage = (
            has_upsampler
            and os.environ.get(
                "LTX2_TWO_STAGE", "0") == "1"
        )

        _log_gpu_memory("denoising-start")

        if use_two_stage:
            # ─── Two-stage pipeline (experimental) ────────
            logger.info(
                "[LTX2-Distilled] Running two-stage pipeline "
                "(LTX2_TWO_STAGE=1)")
            self._run_two_stage(
                batch=batch,
                prompt_embeds=prompt_embeds,
                target_dtype=target_dtype,
                autocast_enabled=autocast_enabled,
                fastvideo_args=fastvideo_args,
            )
        else:
            # ─── Single-stage at full res (default) ───────
            if not has_upsampler:
                logger.info(
                    "[LTX2-Distilled] Running single-stage "
                    "(no upsampler loaded)")
            else:
                logger.info(
                    "[LTX2-Distilled] Running single-stage "
                    "(set LTX2_TWO_STAGE=1 for two-stage)")
            self._run_single_stage(
                batch=batch,
                prompt_embeds=prompt_embeds,
                target_dtype=target_dtype,
                autocast_enabled=autocast_enabled,
                fastvideo_args=fastvideo_args,
            )

        _log_gpu_memory("denoising-done")
        logger.info("[LTX2-Distilled] Denoising done.")
        return batch

    # ─────────────────────────────────────────────────────────
    # Two-stage implementation
    # ─────────────────────────────────────────────────────────

    def _run_two_stage(
        self,
        batch: ForwardBatch,
        prompt_embeds: torch.Tensor,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
        fastvideo_args: FastVideoArgs,
    ) -> None:
        """Execute the full two-stage distilled pipeline."""
        full_latents = batch.latents
        assert full_latents is not None
        device = full_latents.device

        # ── Stage 1: half-resolution generation ──────────
        # Create half-res noise latents
        b, c, f, h, w = full_latents.shape
        half_h, half_w = h // 2, w // 2

        logger.info(
            "[LTX2-Distilled] Stage 1: half-res %dx%d "
            "(%d steps)",
            half_h, half_w, len(DISTILLED_SIGMA_VALUES) - 1,
        )

        # Generate half-res noise.
        # The batch generator may be on CPU; for CUDA tensors
        # we create a device-local generator with the same seed.
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

        stage_1_sigmas = torch.tensor(
            DISTILLED_SIGMA_VALUES,
            device=device,
            dtype=torch.float32,
        )

        # Init audio latents for stage 1
        (audio_latents, audio_context,
         audio_timestep_tpl) = self._init_audio_latents(
            batch, half_latents, fastvideo_args,
            noise_scale=1.0,
        )

        # Temporarily update batch for half-res
        orig_height = batch.height
        orig_width = batch.width
        spatial_ratio = (
            fastvideo_args.pipeline_config.vae_config
            .arch_config.spatial_compression_ratio)
        batch.height = half_h * spatial_ratio
        batch.width = half_w * spatial_ratio

        half_latents, audio_latents = self._denoise_loop(
            latents=half_latents,
            prompt_embeds=prompt_embeds,
            sigmas=stage_1_sigmas,
            audio_latents=audio_latents,
            audio_context=audio_context,
            audio_timestep_template=audio_timestep_tpl,
            target_dtype=target_dtype,
            autocast_enabled=autocast_enabled,
            fastvideo_args=fastvideo_args,
            batch=batch,
            stage_label="Stage 1 (half-res)",
        )

        # Restore original dimensions
        batch.height = orig_height
        batch.width = orig_width

        _debug_save_latent(
            half_latents, "stage1_half_res", audio_latents)
        _debug_decode_and_save(
            half_latents, self._debug_vae,
            "stage1_half_res")


        # ── Spatial upsample ─────────────────────────────
        logger.info(
            "[LTX2-Distilled] Upsampling latents 2× "
            "(%dx%d → %dx%d)",
            half_h, half_w, h, w,
        )
        assert self.spatial_upsampler is not None
        assert self.per_channel_statistics is not None

        # Move upsampler to GPU for the upsample step,
        # then offload back to CPU to free VRAM for stage 2.
        # (~1 GB in bf16, loaded lazily to avoid OOM.)
        self.spatial_upsampler.to(device)
        pcs = self.per_channel_statistics.to(device)

        with torch.no_grad():
            upsampled_latents = upsample_video_latent(
                latent=half_latents[:1],
                upsampler=self.spatial_upsampler,
                per_channel_statistics=pcs,
            )

        # Offload upsampler back to CPU to free VRAM
        self.spatial_upsampler.to("cpu")
        self.per_channel_statistics.to("cpu")
        torch.cuda.empty_cache()
        _log_gpu_memory("after-upsample-offload")

        # ── Async Gemma prefetch for next request ────
        # Start moving Gemma to GPU in a background thread
        # while we prepare Stage 2.  By the time the next
        # request's text encoding runs, Gemma is already
        # on GPU — saving ~10 s of CPU→GPU transfer.
        if envs.GEMMA_PREFETCH_MODE == "prefetch_during_upsample":
            self._async_prefetch_gemma(batch, device)

        _debug_save_latent(upsampled_latents, "upsampled")
        _debug_decode_and_save(
            upsampled_latents, self._debug_vae,
            "upsampled")


        # If batch > 1, repeat for all items
        if b > 1:
            upsampled_latents = upsampled_latents.repeat(
                b, 1, 1, 1, 1)

        # ── Stage 2: full-resolution refinement ──────────
        stage_2_sigmas = torch.tensor(
            STAGE_2_DISTILLED_SIGMA_VALUES,
            device=device,
            dtype=torch.float32,
        )

        _log_gpu_memory("before-stage2")
        logger.info(
            "[LTX2-Distilled] Stage 2: full-res %dx%d "
            "(%d steps, start_sigma=%.6f)",
            h, w,
            len(STAGE_2_DISTILLED_SIGMA_VALUES) - 1,
            STAGE_2_DISTILLED_SIGMA_VALUES[0],
        )

        # Noise the upsampled latent to the stage 2 start
        # sigma.  The reference does this via
        # ``noise_video_state`` with ``noise_scale``.
        start_sigma = stage_2_sigmas[0]
        noise = torch.randn_like(upsampled_latents)
        # x_noisy = (1 - sigma) * x_clean + sigma * noise
        stage_2_latents = (
            (1.0 - start_sigma) * upsampled_latents.float()
            + start_sigma * noise.float()
        ).to(upsampled_latents.dtype)

        # Re-init audio for stage 2 using stage 1 audio as
        # initial latent (add noise at start_sigma).
        if audio_latents is not None:
            audio_noise = torch.randn_like(audio_latents)
            audio_latents = (
                (1.0 - start_sigma)
                * audio_latents.float()
                + start_sigma * audio_noise.float()
            ).to(audio_latents.dtype)

        # Rebuild audio timestep template for full-res
        # (audio shape doesn't change with video resolution)
        if audio_latents is not None:
            audio_timestep_tpl = torch.ones(
                (b, audio_latents.shape[2]
                 if audio_latents.dim() == 4
                 else audio_latents.shape[1]),
                device=device,
                dtype=torch.float32,
            )

        _debug_save_latent(
            stage_2_latents, "stage2_noised", audio_latents)
        _debug_decode_and_save(
            stage_2_latents, self._debug_vae,
            "stage2_noised")


        stage_2_latents, audio_latents = self._denoise_loop(
            latents=stage_2_latents,
            prompt_embeds=prompt_embeds,
            sigmas=stage_2_sigmas,
            audio_latents=audio_latents,
            audio_context=audio_context,
            audio_timestep_template=audio_timestep_tpl,
            target_dtype=target_dtype,
            autocast_enabled=autocast_enabled,
            fastvideo_args=fastvideo_args,
            batch=batch,
            stage_label="Stage 2 (full-res)",
        )

        _debug_save_latent(
            stage_2_latents, "stage2_final", audio_latents)
        _debug_decode_and_save(
            stage_2_latents, self._debug_vae,
            "stage2_final")


        batch.latents = stage_2_latents
        batch.extra["ltx2_audio_latents"] = audio_latents

    # ─────────────────────────────────────────────────────────
    # Gemma prefetch helper
    # ─────────────────────────────────────────────────────────

    def _async_prefetch_gemma(
        self,
        batch: ForwardBatch,
        device: torch.device,
    ) -> None:
        """Move Gemma to GPU asynchronously in a background thread.

        This overlaps the CPU→GPU transfer (~5-13 s for 18 GB)
        with Stage 2 preparation and execution, so that the next
        request's text encoding finds Gemma already on GPU.

        The reference is obtained from ``batch.extra["_gemma_model_ref"]``
        which is set by :class:`TextEncodingStage`.
        """
        gemma = batch.extra.get("_gemma_model_ref")
        if gemma is None:
            logger.debug(
                "[Gemma-prefetch] No _gemma_model_ref in batch; "
                "skipping prefetch.")
            return

        # Already on the target device — nothing to do.
        if next(gemma.parameters()).device == device:
            logger.debug(
                "[Gemma-prefetch] Gemma already on %s; "
                "skipping prefetch.", device)
            return

        def _prefetch() -> None:
            try:
                logger.info(
                    "[Gemma-prefetch] Moving Gemma to %s "
                    "in background thread …", device)
                gemma.to(device=device)
                logger.info(
                    "[Gemma-prefetch] Gemma now on %s.",
                    device)
            except Exception:
                logger.warning(
                    "[Gemma-prefetch] Background prefetch "
                    "failed; Gemma will be loaded on next "
                    "forward() call.",
                    exc_info=True,
                )

        thread = threading.Thread(
            target=_prefetch, daemon=True)
        thread.start()
        # Don't join — let it run in background during Stage 2.

    # ─────────────────────────────────────────────────────────
    # Single-stage fallback
    # ─────────────────────────────────────────────────────────

    def _run_single_stage(
        self,
        batch: ForwardBatch,
        prompt_embeds: torch.Tensor,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
        fastvideo_args: FastVideoArgs,
    ) -> None:
        """Fallback: single-stage at full resolution."""
        latents = batch.latents
        assert latents is not None

        sigmas = torch.tensor(
            DISTILLED_SIGMA_VALUES,
            device=latents.device,
            dtype=torch.float32,
        )

        logger.info(
            "[LTX2-Distilled] Single-stage: %d steps, "
            "latents=%s",
            len(sigmas) - 1,
            tuple(latents.shape),
        )

        (audio_latents, audio_context,
         audio_timestep_tpl) = self._init_audio_latents(
            batch, latents, fastvideo_args,
        )

        latents, audio_latents = self._denoise_loop(
            latents=latents,
            prompt_embeds=prompt_embeds,
            sigmas=sigmas,
            audio_latents=audio_latents,
            audio_context=audio_context,
            audio_timestep_template=audio_timestep_tpl,
            target_dtype=target_dtype,
            autocast_enabled=autocast_enabled,
            fastvideo_args=fastvideo_args,
            batch=batch,
            stage_label="single-stage",
        )

        _debug_save_latent(
            latents, "single_stage_final", audio_latents)
        _debug_decode_and_save(
            latents, self._debug_vae,
            "single_stage_final")


        batch.latents = latents
        batch.extra["ltx2_audio_latents"] = audio_latents

    # ─────────────────────────────────────────────────────────
    # Verification
    # ─────────────────────────────────────────────────────────

    def verify_input(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            "latents", batch.latents,
            [V.is_tensor, V.with_dims(5)])
        result.add_check(
            "prompt_embeds", batch.prompt_embeds,
            V.list_not_empty)
        return result
