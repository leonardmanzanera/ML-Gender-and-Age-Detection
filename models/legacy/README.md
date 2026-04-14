# Legacy Models

These models were used in the early versions of AG Vision (V1/V2) for **age estimation via Caffe CNN**. Since V3, age estimation is handled by the **ViT ONNX** model (`vit_age_gender.onnx`) which provides continuous regression instead of discrete bins.

## Contents

| File | Size | Used in | Replaced by |
|---|---|---|---|
| `age_deploy.prototxt` | 2 KB | V1, V2 | ViT ONNX (V3+) |
| `age_net.caffemodel` | ~44 Mo | V1, V2 | ViT ONNX (V3+) |

## Why keep them?

- The Colab notebook (`AG_Vision_Notebook.ipynb`) downloads and loads these files for the **pedagogical comparison** between Caffe binned age and ViT continuous age (Section 3.3).
- Pipelines V1 and V2 (`pipelines/v1_baseline.py`, `pipelines/v2_yolo_caffe.py`) still reference these models.

## Note

These files are **NOT loaded** by any pipeline from V3 onwards. They are preserved here for backward compatibility and educational purposes only.
