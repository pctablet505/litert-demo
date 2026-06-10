# Resume Prompt: Gemma4 Multimodal LiteRT-LM Export

Use this prompt when starting on a **brand new computer/environment** to resume the work from the 2026-06-10 session.

---

## Step 1: Clone Repositories

```bash
cd /content  # or your preferred workspace

# Install gh if not already available
curl -sS https://webi.sh/gh | sh

# Clone keras-hub (multimodal branch)
gh repo clone pctablet505/keras-hub -- --branch multimodal-litertlm-gemma4-export

# Clone demo repos
gh repo clone pctablet505/gemmademo-litertlm-android-app
gh repo clone pctablet505/gemmademo-litert-export
gh repo clone pctablet505/litert-demo
```

---

## Step 2: Install Dependencies

```bash
# Keras from source (master branch)
gh repo clone keras-team/keras keras-team-source
cd keras-team-source && pip install -e . && cd ..

# Keras-hub from source (editable)
cd keras-hub && pip install -e . && cd ..

# LiteRT packages
pip install litert-torch litert-lm-builder

# Verify
python -c "import keras; print(keras.__version__)"
python -c "import keras_hub; print(keras_hub.__version__)"
```

---

## Step 3: Apply Latest Fixes (if not in branch yet)

The branch `multimodal-litertlm-gemma4-export` should already contain the fixes. Verify by checking the latest commit:

```bash
cd keras-hub && git log --oneline -3
```

If the fixes are missing (e.g., commit `9204f89d` is not present), apply them manually:

### Fix A: `keras_hub/src/utils/litertlm/adapter.py`

In `KerasHubLiteRTAdapter.forward_prefill`:
1. Add `pixel_values=None, pixel_position_ids=None` to the signature.
2. Replace the `if self.has_vision and images is not None:` block with:

```python
if self.has_vision:
    vision_encoder = self.keras_model.backbone.vision_encoder
    is_gemma4_vision = (
        hasattr(vision_encoder, "inputs")
        and len(vision_encoder.inputs) == 2
        and {inp.name for inp in vision_encoder.inputs}
        == {"pixel_values", "pixel_position_ids"}
    )
    if is_gemma4_vision:
        if pixel_values is not None and pixel_position_ids is not None:
            img_embeddings = vision_encoder(
                {
                    "pixel_values": pixel_values,
                    "pixel_position_ids": pixel_position_ids,
                }
            )
    elif images is not None:
        img_embeddings = vision_encoder(images)
```

### Fix B: `keras_hub/src/utils/litertlm/export.py`

1. In `export_to_litertlm`, when `has_vision` is true, detect `is_gemma4_vision` (same heuristic) and call either `_build_gemma4_vision_sample_inputs` or `_build_vision_sample_inputs`.

2. Add `_build_gemma4_vision_sample_inputs` function:

```python
def _build_gemma4_vision_sample_inputs(
    batch_size, max_images, patch_size, image_size,
    num_vision_tokens, seq_len,
):
    device = "cpu"
    num_patches = (image_size // patch_size) ** 2
    patch_dim = patch_size * patch_size * 3
    pixel_values = torch.zeros(
        (batch_size, max_images, num_patches, patch_dim),
        dtype=torch.float32, device=device)
    pixel_position_ids = torch.zeros(
        (batch_size, max_images, num_patches, 2),
        dtype=torch.int32, device=device)
    vision_indices = torch.zeros(
        (batch_size, num_vision_tokens), dtype=torch.int32, device=device)
    vision_mask = torch.zeros(
        (batch_size, seq_len), dtype=torch.int32, device=device)
    return {
        "pixel_values": pixel_values,
        "pixel_position_ids": pixel_position_ids,
        "vision_indices": vision_indices,
        "vision_mask": vision_mask,
    }
```

3. Update `_get_audio_config` to return `audio_input_feat_size`:

```python
audio_input_feat_size = getattr(
    preprocessor, "audio_input_feat_size", 128
)
# ... add to returned dict
```

4. Update `_build_audio_sample_inputs` signature to accept `audio_input_feat_size=128` and use it for the last dimension of `audio_mel`.

5. Update `_PrefillAdapter.forward` to accept and forward `pixel_values`/`pixel_position_ids`.

---

## Step 4: Run Dummy Config Test

```bash
export KERAS_BACKEND=torch
cd /content
python test_gemma4_litertlm_export.py
```

If `test_gemma4_litertlm_export.py` does not exist, recreate it from `litert-demo/session-2026-06-10.md` or write a new one that:
- Builds a tiny `Gemma4Backbone` with `vocabulary_size=16`, `image_size=64`, `patch_size=8`, `pool_size=2`, small layer counts
- Wraps it in `Gemma4CausalLM`
- Calls `export_to_litertlm(..., load_weight=False)` with 1 prefill seq_len and 3 decode seq_lens
- Asserts the `.litertlm` file is created

---

## Step 5: Run Existing Tests

```bash
cd /content/keras-hub
python -m pytest keras_hub/src/utils/litertlm/export_test.py -v
```

All 10 tests should pass.

---

## Step 6: Verify Actual Weights

**Important**: This requires significant RAM (~10 GB for 2B model) and a stable internet connection for weight download.

```python
import os
os.environ["KERAS_BACKEND"] = "torch"

from keras_hub import models
from keras_hub.src.utils.litertlm.export import export_to_litertlm

model = models.Gemma4CausalLM.from_preset("gemma4_instruct_2b")
export_to_litertlm(model, "gemma4_2b.litertlm")
```

If download times out, retry with a longer timeout or use `curl`/`wget` to pre-download weights to `~/.cache/keras_hub/`.

---

## Step 7: Android Device Testing

**Do NOT attempt Android emulator** unless you have KVM (`/dev/kvm`). Instead:

### Option A: Physical Device
```bash
# Connect device via USB, enable ADB debugging
adb devices
adb push gemma4_2b.litertlm /data/local/tmp/
# Run via gemmademo-litertlm-android-app or a minimal test binary
```

### Option B: Firebase Test Lab
Upload the `.litertlm` and test APK to Firebase Test Lab.

### Option C: Local Machine with KVM
If you have a Linux machine with KVM:
```bash
emulator -avd test_avd -no-window -no-boot-anim -gpu swiftshader_indirect
```

---

## Current Status at Start of Session

- ✅ Dummy config export works (vision + audio)
- ✅ Existing unit tests pass
- ⏸️ Actual weight verification pending (download timed out)
- ⏸️ On-device testing pending (no KVM in prior environment)

## Known Blockers to Watch For

1. **Image preprocessing**: `ops.image.resize` with `bicubic` + `antialias=True` is not supported by `litert_torch`. Preprocess images to `pixel_values` + `pixel_position_ids` outside the model graph.
2. **Audio preprocessing**: Verify mel spectrogram generation can happen outside the model or is supported by the converter.
3. **OOM**: Gemma4 2B weights are ~10 GB. Ensure sufficient RAM. Larger variants (4B, 12B, 27B) will need proportionally more.

---

## Files to Check for Context

- `litert-demo/session-2026-06-10.md` — full session log with all details
- `keras_hub/src/utils/litertlm/adapter.py` — PyTorch adapter for LiteRT-LM
- `keras_hub/src/utils/litertlm/export.py` — export orchestration
- `keras_hub/src/utils/litertlm/export_test.py` — unit tests
- `gemmademo-litert-export` — export scripts and examples
- `gemmademo-litertlm-android-app` — Android app consuming `.litertlm`
