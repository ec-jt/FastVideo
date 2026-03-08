# LTX-2.3 Multi-Prompt Audio Quality Test Plan

## Goal

Test the V2 feature extractor normalization fix across diverse prompts
on the LTX-2.3 distilled two-stage pipeline, evaluating both video and
audio content quality. Also implement the missing `enhance_prompt`
function from the upstream Gemma text encoder.

## Status

- V2 normalization fix: ✅ Implemented and verified
- Distilled pipeline: ✅ Working (single-stage and two-stage)
- Dev one-stage pipeline: ✅ Working
- Enhance prompt: ❌ Not yet implemented in FastVideo

## Part 1: Implement Enhance Prompt

The upstream LTX-2 repo uses Gemma3 `generate()` with a system prompt
to expand short user prompts into detailed video+audio descriptions.
FastVideo currently has no equivalent.

### How It Works Upstream

1. Load the Gemma3 model (already loaded for text encoding)
2. Format a chat message with a system prompt + user prompt
3. Call `model.generate()` with `max_new_tokens=512`
4. Clean the response (remove curly quotes, leading non-letters)
5. Use the enhanced prompt for text encoding

### Files to Modify

| File | Change |
|------|--------|
| `fastvideo/models/encoders/gemma.py` | Add `enhance_t2v()` method to `LTX2GemmaTextEncoderModel` |
| `fastvideo/pipelines/stages/text_encoding.py` | Add optional enhance_prompt flag |
| `assets/prompts/gemma_t2v_system_prompt.txt` | Copy system prompt from upstream |
| `fastvideo/entrypoints/video_generator.py` | Add `enhance_prompt` parameter |

### Implementation Details

```python
# In LTX2GemmaTextEncoderModel:
def enhance_t2v(
    self,
    prompt: str,
    max_new_tokens: int = 512,
    seed: int = 10,
) -> str:
    model = self.gemma_model
    tokenizer = AutoTokenizer.from_pretrained(self.gemma_model_path)
    messages = [
        {"role": "system", "content": self._t2v_system_prompt},
        {"role": "user", "content": f"user prompt: {prompt}"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)
    torch.manual_seed(seed)
    outputs = model.generate(
        inputs, max_new_tokens=max_new_tokens, do_sample=True
    )
    return _clean_response(
        tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    )
```

## Part 2: Multi-Prompt Test Suite

### Test Prompts

Diverse prompts covering different audio scenarios, following the
LTX-2.3 prompting guide (detailed, chronological, single paragraph,
present tense, integrated audio descriptions).

#### 1. Music Performance (Jazz)
> A warm, intimate jazz club with dim amber lighting. A saxophone player
> in a dark suit stands center stage, eyes closed, playing a smooth
> melodic solo. Behind him, a pianist gently comps chords on a grand
> piano while a drummer brushes the snare with wire brushes. The rich
> sound of the saxophone fills the room, accompanied by soft piano
> chords and the gentle swish of brushes on the snare. The camera slowly
> dollies in from a medium shot to a close-up of the saxophonists
> fingers moving over the keys.

#### 2. Nature Ambience (Forest)
> A serene forest clearing at golden hour. Tall pine trees surround a
> small stream that trickles over smooth rocks. Birds sing in the
> canopy above — a robin's clear whistle followed by a woodpecker's
> rhythmic tapping. The gentle sound of water flowing over stones mixes
> with rustling leaves as a light breeze passes through. A deer steps
> cautiously into the clearing, its hooves crunching softly on dry
> leaves. The camera holds a wide establishing shot, slowly panning
> right to follow the deer.

#### 3. Urban Street Scene (City)
> A busy New York City intersection at midday. Yellow taxis honk as
> they navigate through traffic. Pedestrians cross the street in a
> hurried pace, their footsteps creating a constant rhythm on the
> asphalt. A street musician plays acoustic guitar on the corner,
> strumming a folk melody that cuts through the urban noise. The camera
> tracks a woman in a red coat as she walks past the musician, pausing
> briefly to listen before continuing. Car engines rumble, a distant
> siren wails, and the guitarist's clear notes ring out.

#### 4. Dialogue Scene (Conversation)
> A cozy kitchen with warm morning light streaming through the window.
> A woman in her 30s with short brown hair sits at a wooden table,
> holding a coffee mug. She looks up and says in a cheerful voice,
> "Good morning! Did you sleep well?" A man enters the frame from the
> right, yawning and stretching. He replies in a groggy voice, "Not
> really, the neighbors dog was barking all night." The sound of coffee
> being poured fills the background as the woman stands and walks to
> the counter. Soft ambient kitchen sounds — the hum of a refrigerator,
> a clock ticking on the wall.

#### 5. Action Scene (Explosion)
> A dusty desert road stretches into the distance under a harsh midday
> sun. A black SUV races toward the camera at high speed, kicking up a
> massive cloud of dust. The roar of the engine grows louder as it
> approaches. Suddenly, the vehicle swerves hard to the left as an
> explosion erupts on the road ahead — a thunderous boom followed by
> debris flying through the air. The camera shakes from the blast wave.
> The SUV skids to a stop, tires screeching on gravel. Smoke and dust
> billow across the frame as the engine idles.

#### 6. Singing/Music (Vocal)
> A young woman with long dark hair stands alone on a dimly lit stage,
> a single spotlight illuminating her face. She holds a microphone close
> and begins singing softly in a clear soprano voice, "Somewhere over
> the rainbow, way up high." Her voice echoes gently in the empty
> concert hall. The camera slowly pushes in from a medium shot as she
> closes her eyes and the melody builds. Faint reverb fills the space.
> She sways slightly, her voice growing stronger with each phrase.

#### 7. Mechanical/Industrial (Factory)
> Inside a large industrial factory, heavy machinery operates in a
> rhythmic pattern. Metal presses stamp down with loud clanging sounds
> at regular intervals. Sparks fly from a welding station in the
> background, creating bright orange flashes. A worker in a hard hat
> and safety goggles walks along the production line, inspecting parts.
> The constant hum of conveyor belts mixes with the sharp hiss of
> pneumatic tools. The camera tracks the worker from a low angle as
> fluorescent lights flicker overhead.

#### 8. Water/Ocean (Waves)
> A dramatic coastal cliff at sunset. Massive ocean waves crash against
> the dark rocks below, sending white spray high into the air. The deep
> rumble of the ocean fills the soundscape, punctuated by the sharp
> crack of each wave impact. Seagulls cry overhead, circling in the
> golden light. The camera holds a wide shot from the cliff edge,
> looking down at the churning water. Wind whistles past, and the
> distant sound of a foghorn echoes across the bay.

### Test Configuration

| Parameter | Value |
|-----------|-------|
| Model | LTX2.3-Distilled-Diffusers |
| Pipeline | LTX23DistilledPipeline (two-stage) |
| Frames | 241 (~10 seconds) |
| Resolution | 512×768 |
| Steps | 8 (stage 1) + 3 (stage 2) |
| Guidance | 1.0 (no CFG) |
| Seed | 42 |
| GPUs | 8 × TP |
| Env | LTX2_TWO_STAGE=1 |

### Test Script

Create `tests/helix/test_ltx2_audio_quality.py` that:
1. Loads the distilled model once
2. Runs each prompt sequentially
3. Saves output videos with descriptive filenames
4. Logs generation time and memory per prompt
5. Optionally runs with `enhance_prompt=True` for comparison

### Evaluation Criteria

For each output video, evaluate:
- **Audio content match**: Does the audio match the described sounds?
- **Audio-video sync**: Are sounds synchronized with visual events?
- **Audio quality**: Is the audio clean (no buzzing, artifacts)?
- **Video quality**: Does the video match the prompt description?
- **Dialogue clarity**: For speech prompts, is dialogue intelligible?

## Part 3: Execution Order

1. Implement `enhance_t2v()` in Gemma encoder
2. Create the multi-prompt test script
3. Run all 8 prompts WITHOUT enhance (raw prompts)
4. Run all 8 prompts WITH enhance (enhanced prompts)
5. Compare results and document findings
