# Convolutional autoencoder: first baseline

## Experiment
- Run: ae_20261003_121002_827675
- Dataset: MVTec AD, bottle
- Training: 167 normal images
- Validation: 42 normal images
- Input: RGB, 256 x 256, values in [0, 1]
- Parameters: 214819
- Training duration: 20 epochs
- Best checkpoint: epoch 19
- Best validation MSE: 0.0022631285135589893
- Training and inspection device: NVIDIA GeForce RTX 5080, CUDA

## Observations
Training and validation reconstruction errors decrease and then plateau.
There is no sustained divergence between the two curves.

The four inspected validation reconstructions preserve the background
and coarse bottle outline, but lose fine rings, edges and reflections.
Reconstruction errors occur on normal bottle structures.

Low reconstruction MSE does not establish defect-detection performance.
Detection and localization metrics have not yet been evaluated.

## Next step
Define an image-level anomaly score and a threshold using normal
validation images before evaluating the official test set.

The validation set was also used for checkpoint selection.
Its threshold-calibration results will be exploratory, not an
independent estimate of the false-positive rate.

## Reference
Bergmann et al., Improving Unsupervised Defect Segmentation by
Applying Structural Similarity to Autoencoders.
https://arxiv.org/abs/1807.02011
