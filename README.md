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
cd backend
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

The interactive API documentation is available at `http://localhost:8000/docs`.
The frontend's Vite development origin (`http://localhost:5173`) is allowed by
default. Set `MEDSCAN_CORS_ORIGINS` to a comma-separated list to change it.

## Model setup

Place the exported Keras models at `backend/models/skin.keras` and
`backend/models/cancer.keras`, or set `MEDSCAN_SKIN_MODEL_PATH` and
`MEDSCAN_CANCER_MODEL_PATH` to their locations. TensorFlow is an optional runtime
dependency and must be installed in the backend environment to load these
artifacts.

For each model, set `MEDSCAN_SKIN_CLASSES` or `MEDSCAN_CANCER_CLASSES` to a JSON
array of labels in the exact output-index order used during training. Also set
`MEDSCAN_SKIN_PREPROCESSING` and `MEDSCAN_CANCER_PREPROCESSING` to the model's
training input convention: `zero_one`, `minus_one_one`, or `raw_0_255`. The API
reports models as unavailable and returns HTTP 503 for their prediction routes
until the artifact and matching configuration are present; it never substitutes
demo predictions.

Grad-CAM is not yet connected to the Keras adapter. The final model files and
their target convolution layers are needed to implement and verify faithful
heatmaps. Until then, the response includes `"gradcam": null`.

## Test the backend

```sh
cd backend
python -m pytest
```

## Next integration objectives

- Obtain both exported model files, class-index mappings, preprocessing details,
  and Grad-CAM target-layer names from the model owner.
- Load and validate each model at startup; then verify predictions and
  explainability against representative images.
- Run the Vue app against the API and verify both domains, upload failures, and
  the rendered probability/Grad-CAM states.
- Measure the project acceptance targets (including under-2-second inference)
  on the agreed test hardware; targets are not claimed as achieved yet.