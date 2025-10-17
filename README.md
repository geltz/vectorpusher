# vectorpusher

A single-slider [ComfyUI](https://github.com/comfyanonymous/ComfyUI) conditioning node that gently reshapes CLIP token embeddings before encoding, improving prompt adherence while preserving identity. For each token, the node looks up semantically close neighbors in the CLIP embedding table, computes a safe direction on the local manifold, and nudges the token a small, angle-capped amount. The slider (sculpt_strength) drives a schedule for the neighborhood size, temperature, step size, and angle cap. A small boost is applied to the “g” branch to better capture global style cues. Special tokens are skipped.

## Install

Clone this repo into your `custom_nodes` directory.
```bash
git clone https://github.com/geltz/vectorpusher
```
Restart [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

## Math

Let W be the CLIP token embedding table (rows are tokens). Let w0 be the original embedding vector for a token. Normalize rows and the token vector:

* Wn = rowwise_normalize(W)
* w0n = normalize(w0) 

Compute similarities s = Wn · w0n and select the top k entries (excluding the self-match). Let vals be the selected similarity scores and idx their indices. Before softmax, apply winsorization: clip z = tau * vals to median ± c * MAD, then set alpha = softmax(z). This yields robust neighbor weights. 

Confidence from entropy: H = −sum(alpha * log(alpha)). Define conf = 1 − H / log(k). When the neighbor set agrees (small H), conf is high; when it disagrees, conf is low. 

Attention gating: map the token’s attention to a scalar g in [0, 1]. The effective step multiplier becomes (0.5 + 0.5 * g) * conf. 

Trust-region direction on the local manifold:

1. Target direction t_raw = sum_j alpha_j * Wn[idx_j].
2. Project to the tangent plane at w0n: t = t_raw − (t_raw · w0n) * w0n.
3. If t is near zero, stop; else normalize t.
4. Propose w_star = w0 + step_scale * norm(w0) * t.
5. Spherical angle cap: if angle(w0, w_star) exceeds theta_max_deg, rescale the update so the angle equals theta_max_deg.
   Finally, renormalize to the original norm. This preserves magnitude while constraining direction. 

Multi-scale K blending: choose ks = {round(0.5k), k, round(1.5k)} and taus = {0.9 tau, tau, 1.1 tau}. For each scale i, compute the trust-region step with step_scale divided by the number of scales, then sum the deltas: w_prop = w0 + sum_i (w_star_i − w0). This mixes fine and coarse neighborhoods. 

KL-budgeted step sizing: form a baseline neighbor distribution alpha0 using (k, tau) at w0. For the proposal w_prop, compute beta = softmax(tau * (Wn[idx0] · normalize(w_prop))). If KL(alpha0 || beta) exceeds kappa, shrink the delta and retry a few times. This keeps the local neighbor distribution stable and limits semantic drift. 

Schedule driven by the single slider s in [0, 1]:

* k = round(8 + 12 s)
* tau = 6 + 6 s
* step_scale = 0.06 + 0.14 s
* theta_max_deg = 8 + 12 s
  These values control neighborhood size, softness of weights, step size, and the angular trust region. The “g” branch uses s scaled up to 1.3x (clamped at 1).
  
Implementation notes: embeddings and norms are computed on the active torch device; special tokens are ignored; final tokens are fed into the standard CLIP encode routine to produce conditioning and pooled outputs. The node exposes only one input slider (sculpt_strength).

## Tips

* Higher strength can improve adherence to “nearby” concepts but may reduce prompt contrast. If results drift, lower the slider.
* Designed to be **fast**: Top-K uses `torch.topk`; embeddings are normalized once per pass. 
* Token IDs like BOS/EOS/PAD are skipped automatically. 

## Credits

* **Inspiration:** [Extraltodeus/Vector_Sculptor_ComfyUI](https://github.com/Extraltodeus/Vector_Sculptor_ComfyUI)




