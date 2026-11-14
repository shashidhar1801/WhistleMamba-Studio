# WhistleMamba whistle-contour detector

The Windows package includes the CSR01 joint BASE epoch-12 inference weights. The detector processes four-second windows and returns time in seconds and frequency in hertz. The default learned tracker links predictions into candidate contours. The heuristic tracker is also available.

At 96 kHz, Studio maps the recording onto the model input grid and restricts detection to 4–48 kHz. The released model uses its original weights and normalization in both input modes.

See the [inference settings](app/whistlemamba/resources/csr01_inference.json), [model identity](app/whistlemamba/resources/csr01_identity.json), [weight checksum](model/weights.json) and [model license](model/MODEL_WEIGHTS_LICENSE.txt). Follow [Start Here](START%20HERE.txt) to load the model on Windows.
