# Getting the dataset

This project targets the **Construction Site Safety Image Dataset**
(2,801 images, 10 classes: Hardhat, Mask, NO-Hardhat, NO-Mask,
NO-Safety Vest, Person, Safety Cone, Safety Vest, machinery, vehicle),
in YOLO annotation format.

It isn't bundled in this repo — it's a few hundred MB of images, too
large for a git repo, and it's distributed via Roboflow rather than as
raw files, so there's no single direct-download URL to script around.

**To get it:**
1. Search "Construction Site Safety Image Dataset Roboflow" (it's
   listed under the `sujaykapadnis` / `roboflow-100` public workspace).
2. Export in **YOLOv8 format** (annotations as `.txt` files matching
   `edgeguard::parseYoloLabelFile`'s expected format — see
   `include/edgeguard/labels.hpp`).
3. Unzip into `data/` at the repo root, so you end up with:
   ```
   data/
     train/images/*.jpg   train/labels/*.txt
     valid/images/*.jpg   valid/labels/*.txt
     test/images/*.jpg    test/labels/*.txt
   ```
   (`data/` is gitignored — this layout is what `scripts/train_and_export.py`
   and the evaluation tooling expect.)

## Training and exporting a model

This C++ project is the **inference/deployment** side. Training itself
uses `ultralytics` (Python), which needs a real PyTorch install —
run `scripts/train_and_export.py` on a machine that has one (this repo's
own CI/sandbox environment intentionally doesn't, since the whole point
of this project is proving out CPU-only, no-GPU, no-Python **deployment**,
not training infrastructure). See that script for exact usage.

Once you have `ppe.onnx`, drop it in `models/` and point `edgeguard` /
`edgeguard_bench` at it with `--model=models/ppe.onnx`.
