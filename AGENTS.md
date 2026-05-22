# ADN — Asymmetry Disentanglement Network

MICCAI 2022 — Acute ischemic stroke infarct segmentation in non-contrast CT scans.

## Quick navigation

- **Paper**: https://arxiv.org/pdf/2206.15445.pdf
- **Contact**: homerhm.ni@gmail.com (open an issue)
- **License**: MIT

## Entrypoints

| Task | File | Description |
|---|---|---|
| Train alignment | `train_align_model.py` | Train transformation network T |
| Train main | `train.py` | Jointly train D (asymmetry) + F (segmentation) after T is frozen |
| Test main | `test.py` | Run inference with D+F |
| Test alignment | `test_align_model.py` | Run inference with T only |
| Quick start T | `quick_start1.py` | Toy example training T |
| Quick start D+F | `quick_start2.py` | Toy example training D+F |

## Architecture (3 networks)

1. **T — Transformation Network** (`model/transform_net.py::PlaneFinder`) — learns to find symmetric plane in 3D head CT. Encoder3D → 6-dof pose → spatial transformer.
2. **D — Asymmetry Extraction** (`ResidualUNet3D in_channels=1, out_channels=1, f_maps=32, use_transconv=False, use_dp=True, p=0.2`)
3. **F — Segmentation** (same `ResidualUNet3D` architecture as D)

Training flow: train T first → freeze T → joint train D+F (two-stage: warm_start=1 then warm_start=0).

**unet3d/** (`model/unet3d/`) is borrowed verbatim from [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet/tree/master/pytorch3dunet/unet3d). Do not modify unless syncing upstream.

## Dependencies

Python 3.7.10, PyTorch 1.10.2. No `requirements.txt` or `pyproject.toml`. Key packages:
`torch`, `nibabel`, `numpy`, `torchvision`, `matplotlib`, `scipy`, `imageio`, `opencv-python`, `tqdm`, `Pillow`.

## Data conventions

- **Format**: NIfTI (`.nii`) loaded via `nibabel.load().get_fdata()`
- **Layout**: `{data_dir}/{patient_id}/CT.nii` and `{data_dir}/{patient_id}/mask.nii`
- **Training list**: plain text file listing patient IDs (one per line, whitespace-split)
- **Input shape**: `(B, 1, 40, 256, 256)` — (batch, channel, slices, height, width). CT values normalized `x/255.0`.
- **Labels**: 0=background, 1/2/3/5=infarct (remapped to 1), 4=useless (ignored). Mask divided by 5.0 internally.
- **Augmentation**: random gamma jitter (`gamma=64/255`), random mirror, random rotation (±10°), optional vertical flip.

## Training details

### Step 1 — Train T
```
python train_align_model.py [--gpu "0"] [--batch-size 40] [--learning-rate 1e-5]
```
- Default B=40, lr=1e-5, AdamW (betas=0.9/0.999, wd=5e-4)
- Loss = photometric_flip + photometric_reconstruction
- Saves checkpoints every 1/5 of total steps
- Model checkpoint key: `'state_dict'`

### Step 2 — Train D+F
```
python train.py [--gpu "2,3,4,5,6,7"] [--batch-size 6] [--learning-rate 1e-4]
```
- Requires pretrained checkpoints: `ALIGN_RESTORE_FROM` (T model), `GWM_SEG_RESTORE_FROM` (tissue segmentation)
- Two loss stages (controlled by `NUM_STEPS_USE_REG`):
  - **warm_start**: reg_loss = BCE + Dice on D output vs ground truth
  - **regularization**: 5-term reg loss (size match, zero-sym, max anatomy, avoid GW, avoid CSF)
- Saves merged checkpoint: `{'seg_state_dict': ..., 'asym_state_dict': ...}`
- **Hardcoded paths**: change `root_dir`, `data_dir`, `train_txt` directly in file or via args.

## Testing
```
python test.py [--gpu "0"] [--batch-size 10] [--restore-from B0006_S050000.pth] [--align-restore-from B0040_S012500.pth]
```
- Saves: per-slice PNG masks, sparse probability maps (`.npz`), visualization composites, and animated GIFs.

## Path quirks

All training/testing scripts have **hardcoded default paths** in module-level constants:
- `/data/StrokeCT/AISD_data_resample` — CT data
- `/data/StrokeCT/aisd_train.txt` — training list
- `/data/StrokeCT/aisd_test.txt` — test list
- `/data/StrokeCT/adn/` — main output dir
- `/data/StrokeCT/align_net/` — T output dir

These must be changed per deployment. There is no config file.

## Checkpoint naming

```
B{batch_size:04d}_S{step:06d}.pth
```
Example: `B0040_S012500.pth` = batch_size=40, step=12500.

## Notable gotchas

- **`map2fig`** (`misc.py`) renders heatmaps via matplotlib at **1000 DPI** — extremely slow. Do not call on large batches.
- **Logger** (`misc.py::Logger`) duplicates stdout to a log file. Calls `print()` route through it automatically after the module-level init.
- **GPU selection** via `os.environ["CUDA_VISIBLE_DEVICES"]` set at module load time. Must match available hardware.
- **Data parallel** for multi-GPU: `nn.DataParallel` wrapped around models in train.py and test.py.
- The default `BATCH_SIZE=6` in train.py assumes 6 GPUs. Adjust for fewer GPUs.
- Label value **4 is ignored** (periventricular white matter hyperintensities / useless).
- The `stn()` function is defined locally in each script (not shared) — identical implementation.
- `cv2` (`import cv2`) is used only in align_net scripts for resize + rotate.

## What this repo does NOT have

- No `requirements.txt` / `pyproject.toml`
- No CI, linter, formatter, type checker, or test framework
- No Dockerfile or environment definition
- No pre-commit hooks or code quality tooling
- No inference-only / deployment script
