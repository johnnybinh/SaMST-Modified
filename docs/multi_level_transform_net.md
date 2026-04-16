# Multi-level `TransformerNet`: 64-D style representation + relu2_2 branch

This note proposes extending the style vector from **32 → 64** and adding a **second transformation stage on VGG `relu2_2` features**, fused into the existing encoder at the **conv3** bottleneck (128 channels, **H/4 × W/4**), which matches the spatial layout of `relu2_2` for the same input resolution.

---

## Design summary

| Piece | Role |
|--------|------|
| **64-D `style_representation`** | One learned vector per style slot (still `style_num + 1` slots in `Style_bank`). |
| **Split** `rep = [rep_low \| rep_high]` | **`rep_low` (32-D)**: drives existing `condition_modulate`, `Dynamic_ConvLayer2`, and `CA_layer` (image-level / residual path). **`rep_high` (32-D)**: drives the new mid-level module only. |
| **Mid-level branch** | Takes **content `relu2_2`** (B, 128, H/4, W/4), applies IN + FiLM from `rep_high` + two3×3 convs (128→128), outputs a **residual** added to the conv3 output `y` before residual blocks. |
| **Who computes `relu2_2`?** | **Outside** `TransformerNet`: training already has frozen `Vgg16`. Pass **`relu2_2`** into `forward` so inference does not silently pull in VGG unless you want that. |

**Why fuse at conv3?** After three stride-2 / encoder stages, the tensor is **128 × H/4 × W/4**, same as VGG16 `relu2_2` for a **single** spatial input size. Channel width **128** matches, so fusion is a simple **additive residual** with no channel projection.

**Double-batch training** (`x.repeat(2,1,1,1)`): content is identical in each pair of rows; **`relu2_2`** can be computed once on the first **`n_batch`** images, then **`relu2_2.repeat(2, 1, 1, 1)`** so each duplicated row gets the same spatial features while **`rep_high` still differs** per row (style id differs).

---

## 1. `networks/transfer_net.py`

### 1.1 Configurable conditioning dimension

Today, `condition_modulate` and `Dynamic_ConvLayer2` hard-code **32**. Thread a **`style_dim`** argument (default **32** for backward compatibility).

**`condition_modulate`**

```python
class condition_modulate(torch.nn.Module):
    def __init__(self, in_channels, style_dim=32):
        super(condition_modulate, self).__init__()
        self.style_dim = style_dim
        self.compress_gamma = torch.nn.Sequential(
            torch.nn.Linear(style_dim, in_channels, bias=False),
            torch.nn.LeakyReLU(0.1, True),
        )
        self.compress_beta = torch.nn.Sequential(
            torch.nn.Linear(style_dim, in_channels, bias=False),
            torch.nn.LeakyReLU(0.1, True),
        )
    # forward unchanged: gamma, beta from representation[:, :style_dim] if caller splits
```

**`Dynamic_ConvLayer2`**

```python
class Dynamic_ConvLayer2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, style_dim=32):
        ...
        self.compress_key = torch.nn.Sequential(
            torch.nn.Linear(style_dim, out_channels * kernel_size * kernel_size, bias=False),
            torch.nn.LeakyReLU(0.1, True),
        )
```

**`ResidualBlock`**: pass **`style_dim=32`** into `Dynamic_ConvLayer2`, `condition_modulate`, and `CA_layer`:

```python
# CA_layer forward uses x[1] with shape (B, style_dim); keep style_dim=32 for rep_low
self.ca = CA_layer(channels_in=style_dim, channels_out=feature_channels, reduction=4)
```

### 1.2 `style_representation`: 64-D, device-safe

Replace hard-coded `.cuda()` with buffer/device pattern so CPU inference works.

```python
class style_representation(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        self.params = nn.Parameter(torch.ones(dim))

    def forward(self):
        z = torch.randn(self.dim, device=self.params.device, dtype=self.params.dtype) * 0.1
        return self.params + z
```

### 1.3 New module: `MidLevelStyleResidual`

Operates on **VGG `relu2_2`** (expect **ImageNet-normalized** features, same convention as `utils.normalize_batch` + `Vgg16`).

```python
class MidLevelStyleResidual(nn.Module):
    """Style transfer at VGG relu2_2 resolution: 128 x H/4 x W/4."""

    def __init__(self, channels=128, style_dim=32):
        super().__init__()
        self.in1 = InstanceNorm2d(channels)
        self.cm1 = condition_modulate(channels, style_dim=style_dim)
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = InstanceNorm2d(channels)
        self.cm2 = condition_modulate(channels, style_dim=style_dim)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.relu = nn.ReLU()

    def forward(self, relu2_2, rep_high):
        x = self.in1(relu2_2)
        x = self.cm1(x, rep_high)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.in2(x)
        x = self.cm2(x, rep_high)
        x = self.relu(x)
        x = self.conv2(x)
        return x
```

Optional: a learnable scalar **`self.fuse_scale = nn.Parameter(torch.tensor(1.0))`** to scale the residual.

### 1.4 `TransformerNet.__init__`

```python
STYLE_DIM_LOW = 32
STYLE_DIM_HIGH = 32
STYLE_DIM_TOTAL = STYLE_DIM_LOW + STYLE_DIM_HIGH  # 64

class TransformerNet(torch.nn.Module):
    def __init__(self, style_num):
        super().__init__()
        self.style_dim_low = STYLE_DIM_LOW
        self.style_dim_high = STYLE_DIM_HIGH

        self.style_bank = Style_bank(style_num, dim=STYLE_DIM_TOTAL)

        # All existing cm* / ResidualBlock use style_dim_low (32) for Linear layers
        self.cm1 = condition_modulate(32, style_dim=STYLE_DIM_LOW)
        self.cm2 = condition_modulate(64, style_dim=STYLE_DIM_LOW)
        self.cm3 = condition_modulate(128, style_dim=STYLE_DIM_LOW)
        # ... same for cm4, cm5; ResidualBlock(..., style_dim=STYLE_DIM_LOW)

        self.mid_level = MidLevelStyleResidual(channels=128, style_dim=STYLE_DIM_HIGH)
        self.mid_fuse_scale = nn.Parameter(torch.tensor(1.0))  # optional
```

Construct **`ResidualBlock`** with an extra argument, e.g. **`style_dim=STYLE_DIM_LOW`**, and thread it into **`Dynamic_ConvLayer2`** and **`condition_modulate`**.

### 1.5 `TransformerNet.forward`

```python
def forward(self, X, style_id, relu2_2=None):
    representation = self.style_bank(style_id)   # (B, 64)
    rep_low = representation[:, : self.style_dim_low]
    rep_high = representation[:, self.style_dim_low :]

    y = self.conv1(X)
    y = self.in1(y)
    y = self.cm1(y, rep_low)
    y = self.relu(y)

    y = self.conv2(y)
    y = self.in2(y)
    y = self.cm2(y, rep_low)
    y = self.relu(y)

    y = self.conv3(y)
    y = self.in3(y)
    y = self.cm3(y, rep_low)
    y = self.relu(y)

    if relu2_2 is not None:
        delta = self.mid_level(relu2_2, rep_high)
        y = y + self.mid_fuse_scale * delta

    y = self.res1(y, rep_low)
    y = self.res2(y, rep_low)
    # ...
    y = self.deconv1(y)
    # ... use rep_low for cm4, cm5    return y, representation
```

**Contract:** if **`relu2_2` is `None`**, behavior matches the old single-level path except that **`rep_high` is unused** (you may want to assert non-None in training).

### 1.6 `Style_bank`

```python
class Style_bank(nn.Module):
    def __init__(self, total_style, dim=64):
        super().__init__()
        self.style_para_list = nn.ModuleList(
            style_representation(dim=dim) for _ in range(total_style + 1)
        )
 # forward: unchanged stacking logic, each z is (dim,)
```

---

## 2. Training: `train_model/train1/train.py` (and train2)

After building **`x`** and **before** `x.repeat`, compute **`relu2_2`** from **normalized** content (same as VGG path for losses).

**Sketch:**

```python
x_dev = x.to(device)
x_norm = utils.normalize_batch(x_dev.clone())  # or normalize before repeat
feat_x = vgg(x_norm)
relu2_2 = feat_x.relu2_2

x = x.repeat(2, 1, 1, 1)
# ...
relu2_2 = relu2_2.repeat(2, 1, 1, 1)  # duplicate along batch for paired AE branch

y, embedding = transformer(x.to(device), style_id=batch_style_id, relu2_2=relu2_2)
```

**Order detail:** `normalize_batch` divides by 255 in-place on the tensor you pass; if you reuse **`x`** for the transformer input, **clone** before normalizing for VGG, e.g. **`utils.normalize_batch(x_dev.clone())`**, so **`X` stays in [0,255]** for `TransformerNet` as today.

---

## 3. Inference: `test_model/test/test.py`

Load **`Vgg16`** (frozen) or reuse a small slice; for each content image:

```python
content_norm = utils.normalize_batch(content_image.clone())
relu2_2 = vgg(content_norm.to(device)).relu2_2
output, embedding = style_model(content_image, style_id=[i], relu2_2=relu2_2)
```

**Checkpoint compatibility:** old weights use **32-D** `style_bank` parameters. Loading into a **64-D** model will **fail** or partially load; plan for **train from scratch** or a small migration script that pads extra dimensions.

---

## 4. Parameter / memory impact (rough)

- **Style bank:** `(style_num + 1) × 32` → **`× 64`** learned scalars (small).
- **Mid-level branch:** four **`condition_modulate(128)`** Linear layers with **32 → 128** + two **128→128** conv stacks (moderate).
- **Training:** one extra **VGG forward** per batch on content if you were not already caching `relu2_2` (you already run VGG on `x1` for content loss—see §5).

---

## 5. Optional: avoid a second VGG pass

Content loss already does **`features_x = vgg(x1.to(device))`**. You can **reuse `features_x.relu2_2`**, expand it to **`2 * n_batch`** with **`repeat`**, and pass that into **`transformer`**, provided **`transformer` is called after** that VGG pass or you refactor order so VGG runs once on the **`n_batch`** content and tensors are aligned. This is a small training-loop refactor, not a model change.

---

## 6. Risks / review points

1. **Normalization:** `relu2_2` must come from the **same** normalization as style/content VGG losses (`utils.normalize_batch`).
2. **Spatial size:** content **`image_size`** must match what you use for VGG (it does if the same tensor is resized/cropped once).
3. **Old checkpoints:** incompatible **`state_dict`** for `style_para_list.*.params` shape `(64)` vs `(32)`.
4. **`style_representation.forward` noise:** still adds Gaussian noise each forward; mid-level sees **noisy `rep_high`** every step—consistent with current design but worth noting for ablations.

---

## 7. Minimal diff checklist

- [ ] `style_representation(dim=64)` + `Style_bank(..., dim=64)`
- [ ] `condition_modulate(..., style_dim=)` and `Dynamic_ConvLayer2(..., style_dim=)`
- [ ] `ResidualBlock`: thread `style_dim`, use **`rep_low`** in `forward`
- [ ] `TransformerNet.forward`: split **`rep_low` / `rep_high`**, add **`MidLevelStyleResidual`**, optional scale
- [ ] `forward(..., relu2_2=None)` signature
- [ ] `train.py`: VGG **`relu2_2`**, **`repeat`** for double batch, **`clone`** before normalize
- [ ] `test.py`: compute **`relu2_2`** per image
- [ ] Retrain or migrate checkpoints

---

*This file is a proposal for review only; it is not applied to the codebase automatically.*
