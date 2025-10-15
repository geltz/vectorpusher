# vectorpusher

A tiny, single-slider CLIP-conditioning node that “nudges” token embeddings toward semantically related neighbors while capping angular drift. Inspired by (but simplified from) [Extraltodeus/Vector_Sculptor_ComfyUI](https://github.com/Extraltodeus/Vector_Sculptor_ComfyUI).

## What

Given your prompt tokens, **vectorpusher** finds the most similar embeddings in the CLIP vocabulary and moves each token a small, bounded step toward a softmax-weighted average of those neighbors—preserving the token’s magnitude to avoid destabilizing downstream layers. 

## Install

1. Create a folder: `ComfyUI/custom_nodes/vectorpusher/`
2. Put your `__init__.py` (the node file) inside it.
3. Start/Reload ComfyUI. The node appears as **“vectorpusher”** under `conditioning`.
   (It exports `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`.) 

## Usage

1. Drop **vectorpusher** before your conditioning hits the sampler.
2. Connect:

   * `clip`: your model’s CLIP
   * `text`: your prompt
   * `sculpt_strength`: 0.0–1.0 (start ~0.6–0.9)
3. Feed the output **Conditioning** to your sampler as usual. The node also returns a short **Params** string. 

## Interface

* **sculpt_strength** *(float, 0→1)* — the only control. Higher values increase neighbor count, sharpen similarity weighting, enlarge the step, and relax the angle cap. Defaults to 1.0 and is clamped in-node per branch logic.  

## Math

* **Top-K neighbors:** For each token vector (w), compute cosine similarity against the CLIP embedding matrix; pick Top-K and softmax-weight them with temperature (\tau). 
* **Orthogonalized target:** Subtract the component of the target along (w) to avoid mere re-scaling.
* **Trust-region step:** Move (w) by a fraction of (|w|) toward that target, then **cap the angle** to a maximum (\theta_{\max}) and **restore (|w|)**. This bounds semantic drift and keeps conditioning stable. 
* **Single slider scheduling:** `sculpt_strength` smoothly sets (K, \tau, \text{step}, \theta_{\max}) via simple monotone schedules. 
* **Branch-aware scaling:** On SDXL, the g-branch gets a mild boost but is hard-clamped to safe range. 

## Values

* Stylized/art prompts: `0.6–1.0`
* Photographic/specific prompts: `0.3–0.6`

## Tips

* Higher strength can improve adherence to “nearby” concepts but may reduce prompt contrast. If results drift, lower the slider.
* Designed to be **fast**: Top-K uses `torch.topk`; embeddings are normalized once per pass. 
* Token IDs like BOS/EOS/PAD are skipped automatically. 

## Credits

* **Inspiration:** [Extraltodeus/Vector_Sculptor_ComfyUI](https://github.com/Extraltodeus/Vector_Sculptor_ComfyUI)

* **This implementation:** minimal, one-knob design and trust-region update derived from the math above.
