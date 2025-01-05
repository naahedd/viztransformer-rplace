# R/Place Vision Transformer Experiment
// Still in progress //
This project explores Vision Transformers (ViTs) for predicting pixel changes on the 2022 Reddit Place canvas. Using a dataset of every pixel change, I’m building models to predict future colors and the time until the next change. It’s still a work in progress, and I’m actively experimenting with architectures and training strategies.

## Dataset
The R/Place dataset is preprocessed into SQLite partitions for faster queries. Two PyTorch datasets are used:
- **ColorDataset**: Predicts the future color of the center pixel.
- **TimeDataset**: Predicts the time until the next pixel change.

## Model
The Vision Transformer combines:
1. **Patch-based Input**: Processes 4x4 canvas patches.
2. **Pixel-based Input**: Handles individual pixels for finer details.

## Training
- **Loss Functions**: CrossEntropyLoss for color, MSELoss for time.
- **Optimizer**: AdamW with a learning rate of 0.001.
- **Hardware**: Supports single or multi-GPU training, though SQLite queries can bottleneck the CPU.
