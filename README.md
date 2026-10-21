# MEDSCAN

MEDSCAN is an educational/research prototype for image classification. It is not
for clinical diagnosis or treatment decisions.

## Project objectives

1. Serve separate skin-disease and skin-cancer classifiers through a documented
   FastAPI service.
2. Validate JPEG/PNG uploads, limit them to 10 MB, decode and preprocess images
   in memory, and return the predicted class and full probability distribution.
3. Add Grad-CAM output when it is compatible with the final trained models.
4. Connect the existing Vue interface to `/health`, `/models`, and the prediction
   routes with consistent error handling.
5. Verify the API contract, invalid-upload behavior, and integrated frontend/API
   flow; document model setup and deployment.

## Current layout

- `backend/` — FastAPI service, model-loading integration, and API tests.
- `Desktop/Medscan-Frontend/` — Vue 3 frontend.

## Start the backend

```sh
source ~/Desktop/LLM/cuda/bin/activate
cd /path/to/Image-classifier-model-main/backend
uvicorn app.main:app --reload
```

Use a Python environment that already has TensorFlow installed to load the
trained models. Install `requirements-dev.txt` separately in a test environment
if pytest is not already available.

The interactive API documentation is available at `http://localhost:8000/docs`.
The frontend's Vite development origin (`http://localhost:5173`) is allowed by
default. Set `MEDSCAN_CORS_ORIGINS` to a comma-separated list to change it.

## Model setup

The uploaded skin model is included at `backend/models/skin.keras`. The backend
loads either a `.keras` archive or an extracted Keras model directory containing
`config.json` and `model.weights.h5`. Place a cancer model at
`backend/models/cancer.keras`, or set `MEDSCAN_SKIN_MODEL_PATH` and
`MEDSCAN_CANCER_MODEL_PATH` to alternate locations. TensorFlow is an optional
runtime dependency and must be installed in the backend environment to load these
artifacts.

The skin model defaults to the HAM10000 output order
`["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]` and MobileNetV2
preprocessing (`minus_one_one`). Override these with `MEDSCAN_SKIN_CLASSES` and
`MEDSCAN_SKIN_PREPROCESSING` if the trained model used different settings. For
the cancer model, set `MEDSCAN_CANCER_CLASSES` to a JSON array of labels in the
exact output-index order used during training and
`MEDSCAN_CANCER_PREPROCESSING` to `zero_one`, `minus_one_one`, or `raw_0_255`.

The backend currently loads the seven-class skin model at `backend/models/skin.keras`
and returns Grad-CAM overlays for it. Set `MEDSCAN_SKIN_GRADCAM_LAYER` or
`MEDSCAN_CANCER_GRADCAM_LAYER` only when the corresponding model needs a layer
other than its final spatial feature layer. Model readiness is reported
independently; the cancer endpoint returns HTTP 503 until a separate cancer
artifact and its class/preprocessing settings are provided. The uploaded `Skin disease.pkl` also serializes a seven-output Keras Functional
model; the API does not deserialize pickle files and instead loads native Keras
artifacts. Both available model artifacts have seven class outputs, so neither
is the separate binary skin-cancer classifier.

## Test the backend

```sh
cd backend
python -m pytest
```

## Remaining integration objectives

- Obtain the separately trained binary skin-cancer artifact, its label order,
  and preprocessing details; the available seven-class model is not a cancer
  classifier.
- Run the Vue app against the API and verify both domains, upload failures, and
  the rendered probability/Grad-CAM states using representative images.
- Measure inference latency on the agreed hardware and representative images;
  the synthetic-input smoke test is not a performance certification.