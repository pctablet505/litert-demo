# LiteRT Export Demo

End-to-end demonstration of exporting KerasHub `Gemma3CausalLM` to LiteRT (`.tflite`) and LiteRT-LM (`.litertlm`) bundles.

## What's Inside

| File | Description |
|------|-------------|
| `slide.md` | 22-slide deep-dive presentation (Markdown + Mermaid) |
| `litert_export_demo.ipynb` | Export to `.tflite` from TF & PyTorch backends + quantization |
| `litertlm_export_demo.ipynb` | Export to `.litertlm` bundle (PyTorch-only) |

## Quick Start

### Environment

- Python 3.10+
- `keras` 3.15+
- `keras-hub` from PR branch `torch-backend-litert-minimal-litertlm`
- `litert-torch`, `litert-lm-builder`, `ai-edge-quantizer`

```bash
pip install keras litert-torch litert-lm-builder ai-edge-quantizer
pip install -e /path/to/keras-hub  # PR #2705 branch
```

### Running the Notebooks

1. **LiteRT Export** (`litert_export_demo.ipynb`)
   - Downloads `hf://google/gemma-3-270m-it`
   - TensorFlow backend export → `gemma3_270m_tf.tflite`
   - PyTorch backend export → `gemma3_270m_torch.tflite`
   - Post-export quantization → `gemma3_270m_torch_wi4afp32.tflite` (~7.7× smaller)
   - Verify with `ai_edge_litert.interpreter.Interpreter`

2. **LiteRT-LM Export** (`litertlm_export_demo.ipynb`)
   - Downloads `hf://google/gemma-3-270m-it`
   - PyTorch backend only
   - Produces `gemma3_270m_it.litertlm` (TFLite + tokenizer + metadata)
   - Verify bundle contents with `litert_lm_builder.litertlm_peek`

> **Note:** Switching `KERAS_BACKEND` between TF and Torch requires a **kernel restart**.

### Android Testing

The `gemmademo-litertlm-android-app` demo app expects `gemma3_270m_it.litertlm` in the app's files directory.

```bash
adb push gemma3_270m_it.litertlm /sdcard/Android/data/com.example.litertlmdemo/files/
```

> **Emulator limitation:** LiteRT-LM currently fails on x86_64 emulators. Use a physical ARM64 device or an ARM64 emulator image.

## Verified Outputs

| Artifact | Size | Backend |
|----------|------|---------|
| `gemma3_270m_tf.tflite` | ~1,073 MB | TensorFlow |
| `gemma3_270m_torch.tflite` | ~1,074 MB | PyTorch |
| `gemma3_270m_torch_wi4afp32.tflite` | ~140 MB | PyTorch + weight-only INT4 |
| `gemma3_270m_it.litertlm` | ~1,083 MB | PyTorch (LiteRT-LM bundle, FP32) |
| `gemma3_270m_it_wi8afp32.litertlm` | ~288 MB | PyTorch (LiteRT-LM bundle, INT8 weights) |

## References

- `keras-team/keras` — Export logic
- `keras-team/keras-hub/pull/2705` — LiteRT-LM export PR
- `google-ai-edge/litert-torch` — PyTorch → LiteRT conversion
- `google-ai-edge/ai-edge-quantizer` — Post-training quantization
