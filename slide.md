# On-Device AI with Keras — LiteRT Export Deep Dive

> Copy-paste ready slide content. Every snippet verified with `keras` master, `keras-hub` master, and `litert-torch` master.

---
## Slide 1: Title

**On-Device AI with Keras**

Supporting model export for LiteRT Runtime

---






## Slide 2: The Problem — Keras Ease vs. Deployment Reality

**Research in Keras is effortless:**

```python
# Keras: 5 lines
model = keras_hub.models.Gemma3CausalLM.from_preset("gemma3_270m")
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.fit(dataset, epochs=3)
model.evaluate(val_dataset)
model.save("my_model.keras")
```

**But deploying on-device is not a "convert" step.** It is a massive engineering overhaul caused by two separate taxes:

1. **Tax 1 — Framework Rewrite:** Re-implement the entire model in pure TensorFlow or PyTorch just to make it exportable.
2. **Tax 2 — Post-Processing:** Rebuild tokenization, sampling, NMS, and rendering in Java/Kotlin/C++ because the runtime only gives you raw tensors.

Same model. Same math. **100× more code, and you pay both taxes on every iteration.**

---






## Slide 3: The Workflow Gap — Research vs. Production

Incompatibility forces a manual bridge between iterative model discovery and optimized deployment.

| **Applied Researchers** | **Production Engineers** |
|---|---|
| **Iterative Experimentation** | **Tax 1: Framework Rewrite** |
| • Tune hyperparameters in Keras | • Architecture study & mapping |
| • Test multiple architectures | • Full re-implementation in TF (300+ lines) |
| • Train on various datasets | • Custom training loop |
| • Select the "Winner" model | • Weight parity verification |
| • `model.generate()` in one line | • Debug numeric divergence |
| • Huge libraries (Keras, HuggingFace) | **Tax 2: Post-Processing Gap** |
| • Unlimited compute resources | • Tokenizer & detokenizer (SentencePiece JNI) |
| | • Sampling strategies (Greedy, Top-K, Top-P) |
| | • NMS, anchor decoding, mask upscaling |
| | • Alpha-blend with camera preview |
| | • Strict constraints (memory, latency, thermal) |
| | • Different languages (C++, Java, Kotlin, Swift) |
| | • Raw tensor I/O only |

**Hand-off: Fragmented Workflow**

→ Weeks of effort lost on every model iteration.

The gap between these worlds is why models die in Jupyter notebooks.

---






## Slide 4: Tax 1 — The Framework Rewrite

```python
# PyTorch research is equally concise
torch_model = GemmaForCausalLM.from_pretrained("google/gemma-3-270m-it")
trainer = Trainer(model=torch_model, args=training_args, train_dataset=ds)
trainer.train()
torch.save(torch_model.state_dict(), "model.pt")
```

**But to deploy on-device, engineering historically had to rewrite everything in pure TensorFlow:**

```python
# Pure TF: 300+ lines of custom training loop
optimizer = tf.optimizers.Adam()
train_loss = tf.keras.metrics.Mean()

def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = compute_loss(logits, y)
        # Custom masking, label smoothing, gradient clipping...
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss.update_state(loss)

# Then: custom checkpointing, metric logging, distributed strategy...
# Then: verify weight loading matches Keras exactly
# Then: export through a separate converter
```

Same model. Same math. **100× more code.**

---






## Slide 5: Tax 1 — It Gets Worse With PyTorch or JAX

The rewrite tax is **not just a Keras → TF problem.**

| Research Stack | On-Device Target | What Engineering Must Do |
|---|---|---|
| **Keras** (any backend) | `.tflite` | Rewrite in pure TF, verify numerics, export |
| **PyTorch** | `.tflite` | Rewrite in pure TF, OR trace → ONNX → TFLite (brittle) |
| **JAX** | `.tflite` | Rewrite in pure TF (no open-source JAX→TFLite converter) |
| **PyTorch** | `.pte` (ExecuTorch) | Rewrite in PyTorch mobile, re-verify, export |

**Every architecture change = restart the rewrite.**

---






## Slide 6: Tax 2 — Post-Processing on Device

Even after the model runs, **raw tensors are useless.** You must rebuild the "last mile" in Java/Kotlin/C++.

**For LLMs:**
- SentencePiece tokenizer (JNI wrapper)
- Padding mask construction
- Sampling: Greedy, Top-K, Top-P, temperature
- KV-cache tensor allocation & indexing
- Detokenization

**For Object Detection:**
- Sigmoid / softmax over class logits
- Confidence thresholding
- Non-Maximum Suppression (NMS) — O(n²) in Java
- Anchor box decoding (center → corners, relative → absolute)
- Scale boxes to original image coordinates

**For Segmentation:**
- Argmax across class dimension
- Upsample mask to original resolution
- Colormap application, alpha-blending with camera preview

**This is not "glue code." It is model-specific, numerically sensitive, and must run at 30 FPS on a thermal-throttled phone.**

---






## Slide 7: Tax 2 — The Post-Processing Gap

```mermaid
flowchart LR
    subgraph DEVICE["On-Device Inference"]
        direction TB
        TFLITE[".tflite Model"] --> INTERP["LiteRT Interpreter"]
        INTERP --> RAW["Raw Output Tensors<br/>logits [1,128,256000]<br/>boxes [1,100,4]<br/>masks [1,512,512,1]"]
        RAW --> POST["<span style='color:#fff;background:#e57373;padding:2px 6px;border-radius:3px'>Post Processing</span>"]
        POST --> FINAL["Final Output<br/>'Hello world'<br/>Labeled boxes<br/>Segmentation overlay"]
    end
    style POST fill:#ef9a9a
    style RAW fill:#fff3e0
```

**With raw LiteRT today, you write all of the red box.**

---






## Slide 8: Alternatives — Domain-Specific Solutions

**OpenCV, MediaPipe, Apple Vision, and QNN reduce post-processing, but they are domain-specific.**

| Alternative | What They Give You | The Limitation |
|---|---|---|
| **OpenCV DNN + NMSBoxes** | Built-in NMS for object detection | Only COCO-style detection. No LLM sampling, no custom heads. |
| **MediaPipe Tasks** | Bundled tokenizer + NMS + rendering | Closed set of Google models. Cannot bring your own architecture. |
| **Apple Vision** | `VNRecognizeTextRequest`, `VNDetectRectangles` | iOS-only, black-box models, no custom weights. |
| **Qualcomm QNN** | SNPE execution + some fused ops | Qualcomm-only. Generative post-processing still manual. |
| **ONNX Runtime Mobile** | Cross-platform inference | Raw tensors only. Tokenization, sampling, NMS still your problem. |

```mermaid
flowchart LR
    subgraph SPECIFIC["Domain-Specific: Works Great"]
        A["MediaPipe Face Mesh"] --> B["Built-in landmarks + rendering"]
        C["OpenCV DNN + NMS"] --> D["Built-in boxes + filtering"]
    end
    subgraph CUSTOM["Your Custom Model: Falls Through"]
        E["Your LLM / Detector / Segmenter"] --> F["❌ No tokenizer<br/>❌ No sampler<br/>❌ No NMS for your anchors<br/>❌ No rendering"]
        style E fill:#fff8e1
        style F fill:#ffcdd2
    end
```

**The moment your problem deviates from their assumptions — custom anchors, new tokenizer, multimodal architecture — you fall back to manual everything.**

---






## Slide 9: The Full Pain — Both Taxes Together

```mermaid
flowchart LR
    subgraph OLD["Old Workflow: Two Taxes"]
        direction TB
        A["Research in Keras / PyTorch / JAX"] --> B["Hand off to Engineering"]
        B --> C["Tax 1: Rewrite in pure TF<br/>300+ lines, weeks"]
        C --> D["Tax 2: Post-processing<br/>Tokenizer + Sampler + NMS + Rendering<br/>200+ lines in Java/Kotlin/C++"]
        D --> E["Weight parity check"]
        E --> F["Retrain if numerics diverge"]
        F --> G["Export toolchain"]
        G --> H["Debug mismatches<br/>Weeks lost"]
        style C fill:#ffcdd2
        style D fill:#ffebee
        style H fill:#ffcdd2
    end
```

**Two taxes. Every model. Every iteration.**

---






## Slide 10: Why We Need Export to LiteRT?

**The KerasHub trap:**
- Currently KerasHub models are only usable through the Python API.
- This makes it almost impossible for on-device AI developers to use the Keras ecosystem end-to-end.
- Even teams that use Keras for R&D have no production deployment path.

**The current dev process:**
1. Train models with Keras
2. Test on desktop
3. Repeat the cycle until reaching a good solution
4. **Then tell the engineering team to write the entire model from scratch in TensorFlow or PyTorch**
5. Convert to device-supported runtime formats (ONNX, TFLite, ExecuTorch, etc.)

**LiteRT export breaks this cycle.**

---






## Slide 11: What is LiteRT?

LiteRT (formerly TensorFlow Lite) is Google's on-device runtime for neural network inference.

- Cross-platform: Android, iOS, Windows, Linux, Embedded, IoT
- Languages: Java, Kotlin, JavaScript, C, C++
- Format: FlatBuffer (`.tflite`) — memory-mapped for instant loading
- Quantization: INT8, FP16, INT4 weight-only
- Hardware: CPU (XNNPACK), GPU (OpenCL/Vulkan), NPU (NNAPI)

---






## Slide 12: The Runtime Landscape

Keras models need to run on diverse high-performance runtimes.

```mermaid
flowchart LR
    K["One Keras Model"] --> L["LiteRT<br/>Android / Google"]
    K --> O["ONNX Runtime<br/>Cross-platform"]
    K --> E["ExecuTorch<br/>PyTorch native"]
    style L fill:#ccffcc
    style K fill:#e1f5fe
```

Our goal: **one Keras model → multiple optimized runtimes** with zero rewrites.

---






## Slide 13: How LiteRT Export Changes Everything

**Old workflow:**

```mermaid
flowchart LR
    A["Research in Keras"] --> B["Finalize architecture"]
    B --> C["Hand off to engineering"]
    C --> D["Tax 1: Rewrite in pure TF"]
    D --> E["Retrain / debug parity"]
    E --> F["Tax 2: Write post-processing<br/>Tokenizer, sampler, NMS, UI"]
    F --> G["Export to .tflite"]
    style C fill:#ffcccc
    style D fill:#ffcccc
    style E fill:#ffcccc
    style F fill:#ffcccc
```

**New workflow:**

```mermaid
flowchart LR
    subgraph NEW["New Workflow"]
        direction TB
        A["Research in Keras<br/>.fit() in 5 lines"] --> B["Finalize architecture"]
        B --> C["model.export<br/>format='litert'"]
        C --> D["One .tflite file"]
        D --> E["Ship to Android"]
        style C fill:#ccffcc
        style D fill:#ccffcc
        style E fill:#ccffcc
    end
```

No rewrite. No custom training loop. No parity debugging.

One line. One model. One flatbuffer.

```python
model.export("model.tflite", format="litert")
```

**Note:** Rewrite is required in PyTorch or TensorFlow; JAX export is not yet open-sourced.

---






## Slide 14: High-Level Flow — Keras to Device

Detailed architecture matching the original PPT Slide 12/13:

```mermaid
flowchart TB
    subgraph DEV["Developer's device"]
        subgraph BACK["Backends"]
            JAX["Jax"]
            TORCH["Torch"]
            TF["Tensorflow"]
        end
        K["Keras"]
        MTS["Model training and<br/>saving"]
        KFILE[".keras file"]
        JAX --> K
        TORCH --> K
        TF --> K
        K --> MTS
        MTS --> KFILE
    end

    subgraph CONV["Converters"]
        AJ["AI-Edge-Jax<br/>❌ Not Open-source"]
        LT["LiteRT-Torch"]
        subgraph TFB["Tensorflow"]
            TLC["tf.lite.TFLiteConverter"]
        end
    end

    KFILE --> LT
    KFILE --> TLC
    KFILE -.-> AJ

    TFLITE[".tflite file"]
    LT --> TFLITE
    TLC --> TFLITE

    subgraph HW["Device / Hardware"]
        subgraph INTERP["Interpreters"]
            ONNX["Onnx<br/>Runtime"]
            LTI["LiteRT/tflite<br/>Interpreter"]
            EXE["ExecuTorch"]
        end
        RAW["Raw Output<br/>Tensors"]
        POST["Post Processing<br/>⚠️ Manual burden"]
        FINAL["Final Output"]
        TFLITE --> LTI
        LTI --> RAW
        ONNX --> RAW
        EXE --> RAW
        RAW --> POST
        POST --> FINAL
    end

    style TFLITE fill:#a5d6a7
    style AJ fill:#ffebee
    style POST fill:#ef9a9a
    style DEV fill:#fafafa
    style HW fill:#fafafa
```

---






## Slide 15: Dual-Backend Export Pipelines

Keras 3 is backend-agnostic. The export API uses different compilers depending on `KERAS_BACKEND`.

### TensorFlow Backend Pipeline

```mermaid
flowchart LR
    subgraph TF["TensorFlow Backend Pipeline"]
        direction TB
        A["Keras Model<br/>(built + trained)"] --> B["ExportArchive.track()"]
        B --> C["add_endpoint('serve',<br/>model.call)"]
        C --> D["TensorFlow SavedModel<br/>(directory)"]
        D --> E["tf.lite.TFLiteConverter<br/>from_saved_model()"]
        E --> F["target_spec:<br/>TFLITE_BUILTINS + SELECT_TF_OPS"]
        F --> G["model.tflite<br/>flatbuffer bytes"]
        style A fill:#e1f5fe
        style D fill:#fff3e0
        style G fill:#ccffcc
    end
```

### PyTorch Backend Pipeline

```mermaid
flowchart LR
    subgraph TORCH["PyTorch Backend Pipeline"]
        direction TB
        A["Keras Model<br/>(torch.nn.Module)"] --> B["Align weights to CPU<br/>device_scope('cpu')"]
        B --> C["torch.export.export()<br/>AOT graph capture"]
        C --> D["ExportedProgram<br/>(ATen dialect ops)"]
        D --> E["litert_torch.convert()"]
        E --> F["ATen → StableHLO<br/>lowering passes"]
        F --> G["LiteRT Compiler<br/>(flatbuffer codegen)"]
        G --> H["model.tflite"]
        style A fill:#e1f5fe
        style D fill:#fff3e0
        style H fill:#ccffcc
    end
```

---






## Slide 16: Verified Export Code — TensorFlow Backend

```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras_hub

model = keras_hub.models.Gemma3CausalLM.from_preset(
    "hf://google/gemma-3-270m-it"
)
model.preprocessor.sequence_length = 128

# Build weights via generate()
model.generate(["What is Keras?"], max_length=16)

# Export — works out of the box
model.export("gemma3_270m_tf.tflite", format="litert")
```

**Notes:**
- The TensorFlow team plans to keep `TFLiteConverter` in the ecosystem. The `Interpreter` API is what shifted to LiteRT.
- TF-backend exports include `Select TF ops` (e.g., `FlexStridedSlice`). On Android you must add `org.tensorflow:tensorflow-lite-select-tf-ops` to your dependencies. PyTorch-backend exports do not need this.

---






## Slide 17: Verified Export Code — PyTorch Backend

```python
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras_hub
from keras import layers

model = keras_hub.models.Gemma3CausalLM.from_preset(
    "hf://google/gemma-3-270m-it"
)
model.preprocessor.sequence_length = 128

# Build via direct call
processed = model.preprocessor({
    "prompts": ["What is Keras?"], "responses": [""]
})
model(processed[0])

# IMPORTANT: explicitly specify input_signature for torch backend
input_signature = [{
    "token_ids": layers.InputSpec(dtype="int32", shape=(1, 128)),
    "padding_mask": layers.InputSpec(dtype="int32", shape=(1, 128)),
}]

model.export(
    "gemma3_270m_torch.tflite",
    format="litert",
    input_signature=input_signature,
)
```

**Why torch backend?**
- Cleaner graph with fewer legacy TensorFlow ops
- Future-proof as Google invests in StableHLO / AI-Edge-Torch
- We also hope to see `ai-edge-jax` open-sourced for JAX backend parity

---






## Slide 18: Quantization — Two Paths, Your Choice

The export API gives you a clean, numerically correct FP32 model. What you do next is your call.

### Path A: Quick Uniform Quantization (via `model.export` kwargs)

```python
import tensorflow as tf

model.export(
    "gemma_quantized.tflite",
    format="litert",
    optimizations=[tf.lite.Optimize.DEFAULT],  # all layers → FP16
)
```

**Best for:** Prototyping, demos, or when you want one precision everywhere.

### Path B: Mixed-Precision (via `ai-edge-quantizer`)

```mermaid
flowchart LR
    subgraph Q["Post-Export Mixed-Precision Quantization"]
        direction TB
        A["Export FP32<br/>.tflite"] --> B["ai-edge-quantizer<br/>Quantizer.load_recipe()"]
        B --> C["Base: dynamic_wi8_afp32<br/>(INT8 weights, FP32 acts)"]
        C --> D["Per-scope overrides:<br/>attention=FP32, FFN=INT4,<br/>embeddings=weight-only"]
        D --> E["Insert Q/DQ nodes<br/>Quantize/Dequantize ops"]
        E --> F["XNNPACK delegates<br/>INT8 GEMM + INT4 weight-only"]
        F --> G["gemma_mixed.tflite<br/>~65MB → ~20MB"]
        style A fill:#e1f5fe
        style G fill:#ccffcc
    end
```

```python
from ai_edge_quantizer import quantizer, recipe

qt = quantizer.Quantizer("gemma_fp32.tflite")
qt.load_quantization_recipe(recipe.dynamic_wi8_afp32())

qt.update_quantization_recipe([
    {"op_type": "FULLY_CONNECTED",
     "scope": ".*attention.*",
     "algorithm_key": "no_quantize"},
    {"op_type": "FULLY_CONNECTED",
     "scope": ".*feedforward.*",
     "algorithm_key": "dynamic_wi4_afp32"},
])

qt.quantize().export_model("gemma_mixed.tflite")
```

**Best for:** Production, where you tune accuracy vs latency per component.

**Why both exist:**
A medical app and a chat demo have different accuracy budgets. A 270M model and a 4B model have different latency constraints. We give you the correct FP32 export; **choosing the quantization strategy is your domain-specific decision**.

---






## Slide 19: What You Bring — Android Application Code

LiteRT gives you a `.tflite` flatbuffer. The rest is your product.

You write:

```kotlin
// 1. Tokenizer (SentencePiece JNI)
val tokens = tokenizer.tokenize(prompt)

// 2. Build padding mask
val paddingMask = IntArray(seqLen) { if (it < tokens.size) 1 else 0 }

// 3. Run inference via Interpreter signature
interpreter.runSignature(inputs, outputs, "serving_default")

// 4. Sample next token from logits
val nextToken = sampler.getNextToken(logits)

// 5. Detokenize
val text = tokenizer.detokenize(generatedTokens)
```

**Why we don't bundle this:**
- Tokenizers vary (SentencePiece, BPE, WordPiece)
- Sampling strategies vary (Greedy for search, Top-P for creative writing)
- UI frameworks vary (Jetpack Compose, React Native, native Canvas)
- We give you the engine. You design the car.

---






## Slide 20: LiteRT-LM Will Eliminate Tax 2 (Generative Models)

LiteRT export eliminates the **model rewrite tax (Tax 1)**.

LiteRT-LM additionally eliminates the **post-processing tax (Tax 2)** for generative models:

| Burden | LiteRT Today | LiteRT-LM Future |
|---|---|---|
| Framework rewrite | ✅ Eliminated via `model.export` | ✅ Eliminated |
| Tokenization | ❌ You bring (SentencePiece JNI) | ✅ Built-in |
| Sampling (Greedy, Top-K, Top-P) | ❌ You write | ✅ Built-in |
| KV-cache management | ❌ Manual tensors | ✅ Automatic |
| Chat templating | ❌ You write | ✅ Runtime applies |

```mermaid
flowchart LR
    subgraph TODAY["Today: LiteRT"]
        direction TB
        K1["Keras Model"] --> E1["model.export<br/>format='litert'"]
        E1 --> R1["Raw Interpreter"]
        R1 --> Y1["You write:<br/>Tokenizer + Sampler + KV Cache + UI"]
        style K1 fill:#fff3e0
        style E1 fill:#fff3e0
        style R1 fill:#fff3e0
    end
    subgraph FUTURE["Future: LiteRT-LM"]
        direction TB
        K2["Keras Model"] --> E2["model.export<br/>format='litertlm'"]
        E2 --> B2[".litertlm Bundle"]
        B2 --> R2["LiteRT-LM Runtime"]
        R2 --> Z1["Built-in:<br/>Tokenizer + Sampler + KV Cache + UI helpers"]
        style K2 fill:#e8f5e9
        style E2 fill:#e8f5e9
        style B2 fill:#ccffcc
        style R2 fill:#ccffcc
    end
```

**Bottom line:** LiteRT export eliminates the **model rewrite tax**. LiteRT-LM will additionally eliminate the **inference boilerplate tax**.

---






## Slide 21: LiteRT vs LiteRT-LM — Clear Distinction

```mermaid
flowchart LR
    subgraph TODAY["Today: LiteRT"]
        direction TB
        K1["Keras Model"] --> E1["model.export<br/>format='litert'"]
        E1 --> R1["Raw Interpreter<br/>(serving_default signature)"]
        R1 --> Y1["You write:<br/>Tokenizer (SentencePiece JNI)"]
        R1 --> Y2["You write:<br/>Sampler (Greedy/Top-K/Top-P)"]
        R1 --> Y3["You write:<br/>KV Cache tensor management"]
        R1 --> Y4["You write:<br/>Jetpack Compose UI"]
        style K1 fill:#fff3e0
        style E1 fill:#fff3e0
        style R1 fill:#fff3e0
    end
    subgraph FUTURE["Future: LiteRT-LM"]
        direction TB
        K2["Keras Model"] --> E2["model.export<br/>format='litertlm'"]
        E2 --> B2[".litertlm Bundle:<br/>TFLite + tokenizer + metadata"]
        B2 --> R2["LiteRT-LM Runtime<br/>(Engine / Conversation)"]
        R2 --> Z1["Built-in:<br/>SentencePiece / BPE tokenizer"]
        R2 --> Z2["Built-in:<br/>Streaming sampler callbacks"]
        R2 --> Z3["Built-in:<br/>KV Cache auto-managed"]
        R2 --> Z4["Built-in:<br/>Chat templating + UI helpers"]
        style K2 fill:#e8f5e9
        style E2 fill:#e8f5e9
        style B2 fill:#ccffcc
        style R2 fill:#ccffcc
    end
```

| | LiteRT (today) | LiteRT-LM (future) |
|---|---|---|
| Export format | `.tflite` | `.litertlm` (bundle) |
| Tokenization | You bring SentencePiece | Built-in (`.spm` or `tokenizer.json`) |
| Sampling | You write Greedy/Top-K/Top-P | Built-in samplers |
| KV Cache | Manual tensor management | Automatic |
| Chat templating | You write | Runtime applies based on model type |
| Speed | Good | Faster — fused prefill/decode kernels |
| Output quality | Good | Better — optimized attention backends |
| Ease of use | Medium | High — drop-in chat generation |
| Status | ✅ Available now | 🔄 PR #2705 — available on the `torch-backend-litert-minimal-litertlm` branch |

**Bottom line:** LiteRT export eliminates the **model rewrite tax**. LiteRT-LM will additionally eliminate the **inference boilerplate tax**.

---






## Slide 22: LiteRT-LM — What's Inside the Bundle

A `.litertlm` file is not just a `.tflite` — it's a **Task Bundle** containing three assets:

```mermaid
flowchart LR
    T["TFLite model<br/>prefill + decode signatures"] --> B["LitertLmFileBuilder"]
    S["Tokenizer asset<br/>.spm or tokenizer.json"] --> B
    M["LlmMetadata .pb<br/>start/stop tokens, context window"] --> B
    B --> O["model.litertlm"]
    style B fill:#ccffcc
    style O fill:#ccffcc
```

**Two signatures inside the TFLite:**

| Signature | Input tokens | Returns | Purpose |
|-----------|-------------|---------|---------|
| **prefill** | Full prompt `seq_len = N` | Updated KV caches only | Fill the cache with the prompt |
| **decode** | Single token `seq_len = 1` | Logits + updated KV caches | Auto-regressive generation |

The runtime calls `prefill` once per turn, then loops on `decode` until a stop token.

---






## Slide 23: LiteRT-LM Export API (PyTorch Backend Only)

```python
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras_hub

model = keras_hub.models.Gemma3CausalLM.from_preset("gemma3_1b")

# Export as .litertlm bundle — PyTorch backend only
# Bucketing: multiple prefill signatures for efficient prompt handling
model.export(
    "model.litertlm",
    format="litertlm",
    prefill_seq_len=[32, 64, 128, 256, 512, 1024],
)
```

**Requirements:**
- `keras.config.backend() == "torch"` — JAX and TensorFlow backends are not supported
- `prefill_seq_len` is baked into the graph; pass a list for bucketing
- **Bucketing increases export time** (~1.5× for 3 buckets) because each signature is traced separately, but model size increase is minimal (~0.07% for 3 extra signatures) since weights are shared
- `quant_config` forwards to `litert_torch.convert()` for in-graph quantization
- For post-export quantization, extract TFLite → `ai-edge-quantizer` → repackage

---






## Slide 24: LiteRT-LM Android Runtime

**Single dependency:**

```kotlin
// build.gradle.kts
dependencies {
    implementation("com.google.ai.edge.litertlm:litertlm-android:0.10.0")
}
```

**Required gradle settings:**

```kotlin
android {
    defaultConfig {
        ndk { abiFilters += listOf("arm64-v8a") }
    }
    androidResources { noCompress += listOf("litertlm") }
    packaging { jniLibs { useLegacyPackaging = true } }
}
```

**Runtime code:**

```kotlin
val engine = Engine(EngineConfig(modelPath, backend = Backend.CPU()))
engine.initialize()
val conversation = engine.createConversation()
conversation.sendMessageAsync("Hello").collect { token ->
    textView.append(token)
}
```

**What the runtime handles for you:**
- Tokenizer invocation (SentencePiece or BytePair)
- Prefill/decode signature dispatch
- KV cache allocation and indexing
- Sampling (Greedy, Top-K, Top-P via `SamplerConfig`)
- Chat templating based on model type (Gemma, Qwen, Llama)
- Streaming tokens via `Flow<Message>`

---






## Slide 25: LiteRT-LM — Current Limitations

| Limitation | Detail |
|------------|--------|
| **PyTorch backend only** | `keras.config.backend() == "torch"`. JAX/TensorFlow not supported. |
| **Fixed prefill length** | `prefill_seq_len` is baked into graph, but **bucketing** (`list[int]`) lets the runtime pick the smallest fitting signature. **Measured 43% faster TTFT** on Pixel 9 for a 31-token prompt vs fixed-128. |
| **Post-export quantization** | FP32/BF16 output. Must use `ai-edge-quantizer` + repackage for deployable sizes. |
| **TFLite 2 GB limit** | Models >~2B params in FP32 exceed FlatBuffer. Must use INT8/INT4. |
| **Emulator unsupported** | Fails on x86_64 emulators. Physical ARM64 devices required. |
| **CPU-only validated** | `Backend.CPU()` works. GPU/NPU delegates exist but untested. |
| **No APK bundling** | Models 275 MB–1 GB+, exceeding Play Store limits. Must push/download at runtime. |
| **First-load latency** | Engine initialization compiles graph on first run (~1–5s on Pixel 9). |
| **Chat template baked** | Developers control content (roles, history) but not the template string. |

---






## Slide 26: LiteRT-LM Before vs After

**Before (raw LiteRT):**

```kotlin
// 200+ lines of InferenceEngine.kt
// Manual tokenization, padding mask, logits sampling,
// KV cache tensor management, signature dispatch
```

**After (LiteRT-LM):**

```kotlin
// ~10 lines
val engine = Engine(EngineConfig(modelPath, backend = Backend.CPU()))
engine.initialize()
val conversation = engine.createConversation()
conversation.sendMessageAsync("What is Keras?")
    .collect { token -> textView.append(token) }
```

---






## Slide 27: Upcoming Challenges & Opportunities

| Item | Status | Path Forward |
|------|--------|--------------|
| `tf.lite.TFLiteConverter` | Staying | TF team confirmed converter remains |
| `tf.lite.Interpreter` | Moved to `ai_edge_litert` | New LiteRT package |
| Torch backend | Available now | `litert_torch` + StableHLO |
| JAX backend | Not yet open sourced | Hope to see `ai-edge-jax` released |
| Mixed-precision in export | User's choice post-export | Use `ai-edge-quantizer` for now |
| LiteRT-LM | PR #2705 (draft) | PyTorch-only, prefill/decode signatures, bundle format |


**Critical challenge from TensorFlow 2.20:**

The `tf.lite` module is deprecated. The existing `tf.lite.TFLiteConverter` API may not be available in the near future.

**Alternative paths forward:**
- Use **AI-Edge-Torch** for PyTorch backend models
- Advocate for **AI-Edge-JAX** to be open-sourced
- Wait for a new converter API in the LiteRT library (not present yet)

*Note: The TensorFlow team has indicated TFLiteConverter will remain in the ecosystem, but the Interpreter API has already shifted to `ai_edge_litert`. Plan for both continuity and migration.*
---






## Slide 28: Production Checklist

- [ ] Choose backend: **TensorFlow** (proven) or **PyTorch** (explicit `input_signature`)
- [ ] Export FP32 `.tflite` and verify with `ai_edge_litert.interpreter.Interpreter`
- [ ] For quick uniform quantization: pass `optimizations=[tf.lite.Optimize.DEFAULT]`
- [ ] For mixed-precision: run `ai-edge-quantizer` with your own component-scoped recipes
- [ ] Write Android code for: tokenizer, padding mask, sampler, detokenizer, UI
- [ ] For LiteRT-LM: ensure PyTorch backend, export `.litertlm`, test on physical ARM64 device
- [ ] Evaluate **LiteRT-LM** when PR #2705 lands — eliminates InferenceEngine boilerplate

---






## Slide 29: References

| Repository | Role |
|-----------|------|
| `keras-team/keras` | Export logic (`keras/src/export/litert.py`) |
| `keras-team/keras-hub` | Gemma presets, tokenizers, multimodal export tests |
| `keras-team/keras-hub/pull/2705` | **LiteRT-LM export PR** — prefill/decode signatures, `.litertlm` bundle (branch: `torch-backend-litert-minimal-litertlm`) |
| `google-ai-edge/litert-torch` | PyTorch → LiteRT conversion stack |
| `google-ai-edge/ai-edge-quantizer` | Post-training mixed-precision quantization |
| `pctablet505/gemmademo-android-app` | Android demo using raw Interpreter |
| `pctablet505/gemmademo-litertlm-android-app` | Android demo using LiteRT-LM runtime |
| `pctablet505/litert-demo` | **This repo** — slides + verified export notebooks |

---






## Slide 30: Thank You

**Questions?**

*Verified companion notebooks:*
- `litert_export_demo.ipynb` — LiteRT (`.tflite`) export from TF & PyTorch backends
- `litertlm_export_demo.ipynb` — LiteRT-LM (`.litertlm`) bundle export

*Landscape diagram PNG exports available in `/content/mermaid_diagrams/`*
