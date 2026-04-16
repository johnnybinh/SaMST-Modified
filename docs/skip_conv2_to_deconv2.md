# Draft: skip connection from encoder `conv2` to decoder `deconv2`

This note specifies a **U-Net-style lateral skip** from the **`conv2` stage** of `TransformerNet` to the **`deconv2` stage** (the **second** upsampling block in the decoder). Goal: pass **H/2 resolution, 64-channel** detail from the encoder into the decoder path that would otherwise only see **H/4** features until after `deconv1`.

**File:** `networks/transfer_net.py`  
**Reference:** current `forward` builds `y` through `conv1` → `conv2` → `conv3` → residuals → `deconv1` → `deconv2` → `deconv3`.

---

## 1. Tensor shapes (assume input `X` is `B × 3 × H × W`)

### Full forward size flow

| Order | Block / tensor | Input size | Output size | Notes |
|-------|----------------|------------|-------------|-------|
| 0 | `X` | `B × 3 × H × W` | `B × 3 × H × W` | Input image |
| 1 | `conv1` | `B × 3 × H × W` | `B × 32 × H × W` | `kernel=9`, `stride=1`, reflection padding keeps resolution |
| 2 | `in1` | `B × 32 × H × W` | `B × 32 × H × W` | No size change |
| 3 | `cm1` | `B × 32 × H × W` | `B × 32 × H × W` | No size change |
| 4 | `ReLU` | `B × 32 × H × W` | `B × 32 × H × W` | No size change |
| 5 | `conv2` | `B × 32 × H × W` | `B × 64 × H/2 × W/2` | Downsample by `stride=2` |
| 6 | `in2` | `B × 64 × H/2 × W/2` | `B × 64 × H/2 × W/2` | No size change |
| 7 | `cm2` | `B × 64 × H/2 × W/2` | `B × 64 × H/2 × W/2` | No size change |
| 8 | `ReLU` | `B × 64 × H/2 × W/2` | `B × 64 × H/2 × W/2` | This is the recommended `skip_conv2` tensor |
| 9 | `conv3` | `B × 64 × H/2 × W/2` | `B × 128 × H/4 × W/4` | Downsample by `stride=2` |
| 10 | `in3` | `B × 128 × H/4 × W/4` | `B × 128 × H/4 × W/4` | No size change |
| 11 | `cm3` | `B × 128 × H/4 × W/4` | `B × 128 × H/4 × W/4` | No size change |
| 12 | `ReLU` | `B × 128 × H/4 × W/4` | `B × 128 × H/4 × W/4` | No size change |
| 13 | `res1` | `B × 128 × H/4 × W/4` | `B × 128 × H/4 × W/4` | Residual block preserves size |
| 14 | `res2` | `B × 128 × H/4 × W/4` | `B × 128 × H/4 × W/4` | Residual block preserves size |
| 15 | `res3` | `B × 128 × H/4 × W/4` | `B × 128 × H/4 × W/4` | Residual block preserves size |
| 16 | `res4` | `B × 128 × H/4 × W/4` | `B × 128 × H/4 × W/4` | Residual block preserves size |
| 17 | `res5` | `B × 128 × H/4 × W/4` | `B × 128 × H/4 × W/4` | Residual block preserves size |
| 18 | `deconv1` | `B × 128 × H/4 × W/4` | `B × 64 × H/2 × W/2` | `Upsample(scale=2)` then `3×3 conv` |
| 19 | `in4` | `B × 64 × H/2 × W/2` | `B × 64 × H/2 × W/2` | No size change |
| 20 | `cm4` | `B × 64 × H/2 × W/2` | `B × 64 × H/2 × W/2` | No size change |
| 21 | `ReLU` | `B × 64 × H/2 × W/2` | `B × 64 × H/2 × W/2` | Natural skip merge point before `deconv2` |
| 22 | `cat([D, S], dim=1)` | `D: B × 64 × H/2 × W/2`, `S: B × 64 × H/2 × W/2` | `B × 128 × H/2 × W/2` | Concatenate decoder + encoder skip |
| 23 | `skip_fuse` (`1×1 conv`) | `B × 128 × H/2 × W/2` | `B × 64 × H/2 × W/2` | Reduce channels back to 64 |
| 24 | `deconv2` | `B × 64 × H/2 × W/2` | `B × 32 × H × W` | Second upsampling stage |
| 25 | `in5` | `B × 32 × H × W` | `B × 32 × H × W` | No size change |
| 26 | `cm5` | `B × 32 × H × W` | `B × 32 × H × W` | No size change |
| 27 | `ReLU` | `B × 32 × H × W` | `B × 32 × H × W` | No size change |
| 28 | `deconv3` | `B × 32 × H × W` | `B × 3 × H × W` | Final RGB output |

### Key sizes for this skip design

| Tensor | Size |
|--------|------|
| `skip_conv2` | `B × 64 × H/2 × W/2` |
| decoder tensor before `deconv2` | `B × 64 × H/2 × W/2` |
| concatenated tensor | `B × 128 × H/2 × W/2` |
| fused tensor passed into `deconv2` | `B × 64 × H/2 × W/2` |
| output of `deconv2` | `B × 32 × H × W` |

So **`skip_conv2`** (encoder) and the tensor **entering `deconv2`** (decoder) already match in **resolution** and **channel width** if you fuse **after** the first decoder upsampling stack and **before** `self.deconv2`.

That is the natural merge point: **one hop before `deconv2`**, not inside `deconv2` itself.

---

## 2. What to save as the skip

**Option A (recommended for a first try):** save activations **after** the same pipeline as today:

```text
skip = conv2 -> in2 -> cm2 -> ReLU   # already style-conditioned at H/2
```

**Pros:** consistent with the rest of the network (same `representation`).  
**Cons:** skip is not “pure content”; it is already modulated by `f`.

**Option B:** save **earlier**, e.g. right after `conv2` and **before** `in2` / `cm2`, or only after `in2` without `cm2`.  
**Pros:** can preserve more structural detail with less style warping on the skip.  
**Cons:** mismatch in statistics vs decoder branch; may need an extra **1×1** or **IN** on the skip for stability.

This draft assumes **Option A** unless you explicitly want a cleaner content lateral.

---

## 3. Fusion rule

Let:

- `S` = `skip_conv2`, shape `B × 64 × H/2 × W/2`
- `D` = decoder tensor after `deconv1` → `in4` → `cm4` → `ReLU`, same shape

**Concatenate** along channels:

```text
F = cat(D, S, dim=1)   # B × 128 × H/2 × W/2
F = Conv1x1(F)         # B × 64 × H/2 × W/2  (learned mixing)
```

Then pass **`F`** into **`deconv2`** (and keep existing `in5`, `cm5`, `ReLU`, `deconv3`).

Add in `__init__`:

```python
self.skip_fuse = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=True)
```

**Alternative:** `F = D + self.skip_proj(S)` with `skip_proj: Conv2d(64, 64, 1)` if you want additive fusion only (fewer channels, sometimes less flexible).

---

## 4. `forward` sketch (conceptual)

```python
def forward(self, X, style_id):
    representation = self.style_bank(style_id)

    y = self.conv1(X)   # B, 32, H, W
    y = self.in1(y)
    y = self.cm1(y, representation)
    y = self.relu(y)

    y = self.conv2(y)   # B, 64, H/2, W/2
    y = self.in2(y)
    y = self.cm2(y, representation)
    y = self.relu(y)
    skip_conv2 = y  # B, 64, H/2, W/2

    y = self.conv3(y)   # B, 128, H/4, W/4
    # ... in3, cm3, relu, res1..res5 ...

    y = self.deconv1(y)  # B, 64, H/2, W/2
    y = self.in4(y)
    y = self.cm4(y, representation)
    y = self.relu(y)

    y = self.skip_fuse(torch.cat([y, skip_conv2], dim=1))  # B, 64, H/2, W/2
    y = self.deconv2(y)  # B, 32, H, W
    y = self.in5(y)
    y = self.cm5(y, representation)
    y = self.relu(y)

    y = self.deconv3(y)  # B, 3, H, W
    return y, representation
```

---

## 5. Alignment pitfalls

1. **Odd `H` or `W`:** strided convs and `Upsample(scale_factor=2)` can produce **off-by-one** spatial sizes between `skip_conv2` and `D`. If `D` and `S` differ by 1 pixel, align with **`F.interpolate`** on the smaller side to match the larger, or crop to a common size (center crop is a simple fix).
2. **Training checkpoints:** adding `skip_fuse` changes `state_dict` keys; old checkpoints **won’t load** without strict=False or a migration that initializes `skip_fuse` only.
3. **Identity branch (`style_id == 0`):** the skip still carries **H/2** detail; verify that reconstruction loss does not regress (may need a small weight on fusion or train slightly longer).

---

## 6. Summary

| Item | Choice in this draft |
|------|----------------------|
| Skip source | Output of **encoder `conv2` stack** (`in2` + `cm2` + ReLU) |
| Merge target | **After `deconv1` stack**, **before `deconv2`** |
| Fusion | **Concat** to128 ch → **`Conv2d(128, 64, 1)`** |
| Name mapping | User “**up2**” ↔ code **`deconv2`** (second upsample in decoder) |

---

## 7. Style-conditioned skip gating (proposal, not yet applied)

### Related work

**CSPANet** — *"Cross-Route Statistical Partition Attention Network for Style Transfer"* (Neurocomputing, 2025) introduces a **Detail-Aware Skip Connection (DSC)** that dynamically gates shallow encoder features before injecting them into the decoder. Our `SkipGate` module follows the same principle.

### Design

Each skip is multiplied by a **per-channel sigmoid gate** derived from the style representation, plus a **manual strength knob**:

```
gated_skip = skip * sigmoid(W @ representation + b) * strength
```

- **Learned gate** (`sigmoid(W @ representation + b)`): each style learns its own per-channel gate. Spatially heavy styles (Van Gogh) can learn to close the gate; color-only styles can keep it open.
- **Manual `strength`** (default `1.0`): a scalar you can override at inference time without retraining. Set to `0.0` to fully disable skips, `0.5` for half strength, etc.

### Usage at inference

```python
model.skip_gate2.strength = 0.3   # dampen H/2 skip (conv2 -> deconv2)
model.skip_gate3.strength = 0.3   # dampen H skip   (conv1 -> deconv3)
```

### Modules added in `__init__`

```python
self.skip_gate2 = SkipGate(channels=64, style_dim=32)   # gates skip_conv2
self.skip_gate3 = SkipGate(channels=32, style_dim=32)   # gates skip_conv1
```

---

## 8. Style-predicted alpha/beta blend (proposal, not yet applied)

### Idea

Instead of fixed scalars (CSPANet) or per-channel sigmoid gates (section 7), predict alpha and beta from the **style representation** using the same MLP pattern already used by `condition_modulate` everywhere in the network.

Replace concat + 1x1 projection with an additive blend:

```
ab = Linear(representation)      # (B, 2)
alpha, beta = ab[:, 0], ab[:, 1]
output = alpha * decoder + beta * skip
```

Each style learns its own content-vs-style ratio. No extra conv layers, no channel doubling.

### Why this fits

`condition_modulate` already does `out = x * gamma + beta` where gamma/beta come from `Linear(representation)`. The skip blend is the same pattern applied to two feature streams instead of one.

### Implementation

```python
# In __init__:
self.skip_blend2 = nn.Linear(32, 2)   # predicts (alpha, beta) for H/2 skip
self.skip_blend3 = nn.Linear(32, 2)   # predicts (alpha, beta) for H skip

# In forward (replaces cat + downsample):
ab = self.skip_blend2(representation)
alpha = ab[:, 0:1].view(-1, 1, 1, 1)
beta  = ab[:, 1:2].view(-1, 1, 1, 1)
y = alpha * y + beta * skip_conv2     # (B, 64, H/2, W/2), no shape change
```

Cost: 66 params per skip level (32x2 weights + 2 bias). Removes `downsample2`/`downsample3` entirely.

### Per-style behavior

| Style type | Expected learned ratio | Effect |
|------------|----------------------|--------|
| Van Gogh (heavy spatial style) | high alpha, low beta | Skip suppressed, spatial style dominates |
| Color grading (no spatial change) | balanced alpha/beta | Skip preserved, sharp output |
| Identity (style_id=0) | low alpha, high beta | Near-passthrough, clean reconstruction |

### Risk: representation capacity

The 32-dim style vector already feeds ~16 linear layers (all `cm` and `Dynamic_ConvLayer2` and `CA_layer` modules). Adding 2 more is a small gradient burden, and skip-blend correlates naturally with style type (not orthogonal information), so it shouldn't destabilize.

**Signs of trouble:** style loss suddenly plateaus or gets worse after adding blend layers.

**Mitigation — staged training:**

1. **Freeze** `skip_blend2` and `skip_blend3` for the first N epochs (e.g. alpha=1, beta=1 fixed). Let the representation stabilize on style first.
2. **Unfreeze** after epoch N. The representation is already good; the blend layers just learn a small correction on top.

This avoids early-training interference where the blend gradients compete with the style/content losses for control of the 32-dim vector.

---

## 9. Summary of options

| Option | Fusion | Style-conditioned | Params | Complexity |
|--------|--------|-------------------|--------|------------|
| Current (concat + 1x1 proj) | Concat then `ConvLayer(2C, C, 1, 1)` | No | ~8K per skip | Moderate |
| Section 7 (SkipGate) | Multiplicative gating before concat | Yes (per-channel sigmoid) | ~4K total | Moderate |
| Section 8 (alpha/beta blend) | Additive weighted sum | Yes (2 scalars from MLP) | ~132 total | Minimal |
| CSPANet DSC (reference) | Additive weighted sum | No (fixed scalars) | 2 total | Trivial |
