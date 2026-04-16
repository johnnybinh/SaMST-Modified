# SaMST paper ↔ code mapping

This document ties the modules and training objective described in **“Pluggable Style Representation Learning for Multi-Style Transfer”** ([arXiv:2503.20368](https://arxiv.org/abs/2503.20368)) to the implementation in this repository. **Depth-based auxiliary losses are omitted here** (they are a local extension, not part of the paper).

Primary implementation file: `networks/transfer_net.py`. Training: `train_model/train1/train.py`, `train_model/train2/train.py`. Perceptual backbone: `loss/vgg.py`. Batch normalization for VGG: `train_model/utils.py` (`normalize_batch`, `gram_matrix`).

---

## 1. Framework overview (paper Section 3.1)

| Paper concept | Role |
|---------------|------|
| Content image **c** | Input RGB image to the transfer network. |
| Encoder | Maps **c** to a spatial feature map **E**. |
| Style codebook (**SCB**) | Holds one compact vector **f_i** per style (and an identity slot for reconstruction). |
| Generator / SaMST blocks | Adapt **E** using **f_i** (local geometry + global statistics + channel gating). |
| Decoder | Maps adapted features back to RGB stylized image **I_i**. |

**Code:** `TransformerNet` is a single U-Net–style hourglass: three encoder conv stages → five residual blocks → two upsampling stages → final 9×9 conv to 3 channels. Conditioning enters via `Style_bank(style_id)` and is threaded through `condition_modulate`, `Dynamic_ConvLayer2`, and `CA_layer` as described below.

---

## 2. Style codebook and style representation (paper Section 3.1, Section 3.2 intro)

**Paper:** Each style is encoded by a compact vector **f_i ∈ ℝ^C** with **C = 16** in the paper’s description. The codebook is *pluggable*: at inference you select **f_i** without growing the core network.

**Code:**

| Component | Location | Behavior |
|-----------|----------|----------|
| **SCB** | `Style_bank` | `nn.ModuleList` of `style_representation`, length **`style_num + 1`**. Indices `1 … style_num` are artistic styles; index **`0`** is the **identity / auto-encoder** slot used when reconstructing the content (paper’s **f_0** and **I_0**). |
| **Learned vector** | `style_representation.params` | A trainable vector of length **32** (not 16). |
| **Stochastic term** | `style_representation.forward` | Adds **Gaussian noise** to `params` each forward pass (`mean=0`, `std=0.1`). The paper describes deterministic style codes from training; this noise is an implementation choice. |

**Forward API:** `TransformerNet.forward(X, style_id)` with `style_id` a per-batch list of integers. `Style_bank` stacks the selected representations into a tensor **`representation`** of shape **(B, 32)** passed to all conditional layers.

**Note:** `Style_bank.forward` also builds `all_z` for every slot but returns only `new_z` for the requested indices; the extra stack is unused (harmless dead computation).

**Incremental style extension (paper):** The paper describes adding new styles without catastrophic forgetting of old ones. **Code:** `Style_bank.add_style(add_num)` sets `requires_grad_(False)` on every existing `style_representation.params`, then appends `add_num` new `style_representation` modules. Training scripts must be updated separately (e.g. style image count, `style_num` passed into `TransformerNet`) so the new slots receive gradients and data.

---

## 3. Encoder and decoder (paper Section 3.1)

**Paper:** Encoder and decoder are symmetric and lightweight (three-level design, comparable to prior MST work).

**Code:**

| Stage | Modules | Channels |
|-------|---------|----------|
| Encoder | `ConvLayer` + `InstanceNorm2d` + `condition_modulate` + ReLU | 3→32 → 32→64 → 64→128 |
| Bottleneck | Five `ResidualBlock`s | 128 throughout |
| Decoder | `UpsampleConvLayer` + `InstanceNorm2d` + `condition_modulate` + ReLU | 128→64 → 64→32 |
| Head | `ConvLayer` 32→3, no norm | RGB output |

Reflection padding is used on convolutions where applicable (`ConvLayer`, `Dynamic_ConvLayer2`, upsampling path).

---

## 4. Style-wise convolution (**SConv**, paper Section 3.2.1)

**Paper:** An MLP maps the style representation **f_i** to depthwise convolution kernels **K_i**. The content feature **E** is updated as **E_out = Sconv(K_i, E)** (depthwise convolution, groups = channels).

**Code:** `Dynamic_ConvLayer2` in `ResidualBlock.conv1`.

- `compress_key`: `Linear(32 → out_channels × k_h × k_w)` + `LeakyReLU`, reshaped to per-batch kernels.
- Kernels are **`repeat_interleave`** along the “input channel per group” dimension so grouped convolution matches channel layout.
- `F.conv2d(..., groups=b * groups)` applies **batched depthwise-style** filtering with different kernels per batch element (per style).

This is the closest match to **SConv**: dynamic kernels entirely determined by **representation**.

---

## 5. Style-representation adaptive instance norm (**SRAdaIN**, paper Section 3.2.2)

**Paper:** MLPs predict scalar **per-channel** **γ_i** and **β_i** from **f_i**, then  
**E_o1 = β_i · (E_out − μ(E_out)) / σ(E_out) + γ_i**.

**Code:** Split across two modules used **in sequence**:

1. **`InstanceNorm2d`** — `affine=False`, so it computes **per-instance** mean/variance normalization (same structural role as **(· − μ) / σ**).
2. **`condition_modulate`** — two MLPs (`compress_gamma`, `compress_beta`) map **representation** to **γ**, **β** with shape **(B, C, 1, 1)** and apply **out = x × γ + β**.

So the code applies **affine modulation after normalization**, whereas the paper writes **β** scaling the normalized map and **γ** as bias. Here **γ** multiplies **x** and **β** is added; naming follows the AdaIN/FiLM literature loosely. Functionally this is still **“normalize then style-dependent scale and shift.”**

`condition_modulate` is used after every `InstanceNorm2d` in the main path and inside each residual branch (see below).

---

## 6. Style-wise channel modulation (**SCM**, paper Section 3.2.3)

**Paper:** MLP + **sigmoid** produces channel coefficients **v_i**; **E_o2 = E * v_i** (element-wise product, broadcast over space). **E_o1** (from SRAdaIN branch) and **E_o2** are **summed** to form the block output.

**Code:** `CA_layer` inside `ResidualBlock`.

- Input: `[residual, representation]` where **residual** is the skip **x** and **representation** is **(B, 32)**.
- `conv_du`: 1×1 convs on **representation** viewed as **(B, 32, 1, 1)** → **sigmoid** → shape **(B, C, 1, 1)** matching feature channels **C = 128**.
- Output: **`residual * att`** (channel-wise gating).

The block sums **`out + self.ca([residual, representation])`** where **`out`** is the processed main branch after the second `condition_modulate`. So the **skip** is modulated in a **SCM-like** way and added to the transformed branch—analogous to injecting **E * v_i** alongside the SConv+SRAdaIN path, though the exact topology differs from the paper figure (sum of two explicit branches vs. residual fusion).

---

## 7. Residual block as a SaMST-style cell (paper Fig. 3(b))

**Paper:** One style-aware block combines **SConv → SRAdaIN → (+ SCM contribution)**.

**Code:** `ResidualBlock.forward`:

1. **`Dynamic_ConvLayer2`** (SConv-like) + IN + `condition_modulate` + ReLU.
2. **`ConvLayer` 1×1** (channel mixing) + IN + `condition_modulate`.
3. **`out + CA_layer([residual, representation])`** — SCM-like gating on the identity skip.

The encoder/decoder stacks outside the residual tower use **fixed** `ConvLayer` + IN + `condition_modulate` only (no `Dynamic_ConvLayer2` there); **dynamic depthwise convolution appears in residuals only**.

---

## 8. Training objective (paper Section 3.3.1) vs code

The paper defines:

**L = λ_c L_c + λ_s L_s + λ_ae L_ae + λ_geo L_geo**

with default weights **λ_c = 1**, **λ_s = 10**, **λ_ae = 0.01**, **λ_geo = 0.01**.

| Term | Paper | This repo |
|------|--------|-----------|
| **L_c** | L2 between VGG16 features of **I_i** and **c** on layer set **{l_c}** | `mse_loss(features_y.relu2_2, features_x.relu2_2)` after `Vgg16` + `normalize_batch` — effectively **one layer** (`relu2_2`). |
| **L_s** | L2 between **μ** and **σ** of VGG features of **I_i** vs style **s_i** on **{l_s}** | **Gram-matrix MSE** over **all** returned layers (`relu1_2` … `relu4_3`) — classic fast neural style, **not** the paper’s moment matching (Eq. (6)). |
| **L_ae** | **‖I_0 − c‖_2** in image space for identity code **f_0** | MSE on **normalized** tensors for the second half of the doubled batch (`y2` vs `x2`) with style id **0** — same *role* (identity reconstruction), different normalization detail. |
| **L_geo** | L1 **equivariance** under a set of spatial transforms **T** (Eq. (8)) | **Not implemented** in training scripts. |

**Training mechanics:** Batches duplicate each content image (`x.repeat(2, 1, 1, 1)`): first half uses a random style id, second half uses **0** for the reconstruction path. See `train_model/train1/train.py` and `train_model/train2/train.py`.

YAML weights in `train_model/train1/train.yml` use large scalars (**1e5**, **1e10**, etc.) tuned for Gram/feature MSE scales, not the paper’s **(1, 10, 0.01, 0.01)**.

---

## 9. Quick reference: symbol → code

| Paper | Code |
|-------|------|
| **f_i**, style code | Row `i` of `Style_bank` → `style_representation()` → **(32,)** |
| **f_0** | `style_id == 0` |
| **E**, content features | Activations inside `TransformerNet` (128 ch at bottleneck) |
| **SConv** | `Dynamic_ConvLayer2` |
| **SRAdaIN** (conceptually) | `InstanceNorm2d` + `condition_modulate` |
| **SCM** (conceptually) | `CA_layer` |
| VGG perceptual loss | `loss/vgg.py` — `Vgg16` |

---

## 10. Summary of intentional / structural differences

1. **Representation dimension:** paper **C = 16**, code **32**.
2. **Style loss:** paper uses **mean/variance** matching; code uses **Gram matrices**.
3. **Geometric loss:** present in the paper, **absent** in repo training.
4. **Block wiring:** paper’s explicit **E_o1 + E_o2** from parallel branches; code uses **residual + channel attention** fusion in `ResidualBlock` and static convs in the outer encoder/decoder.
5. **Noise on style vectors:** used in `style_representation`; not emphasized in the paper’s formalism.

For a separate design note on extending the representation (e.g. 64-D and mid-level VGG fusion), see `docs/multi_level_transform_net.md`.
