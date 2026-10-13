# Facial Emotion Recognition using PyTorch

## Project Overview

This project implements a Facial Emotion Recognition (FER) system using a Convolutional Neural Network (CNN) built with PyTorch. The model classifies facial expressions into seven emotion categories using grayscale facial images from the FER-2013 dataset.

The project covers data preprocessing, custom Dataset and DataLoader implementation, CNN model development, training, evaluation, and prediction on individual test images.

## Dataset

**Dataset Link:** [FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)

The dataset is loaded from `fer2013.csv` using Pandas. The CSV contains emotion labels, pixel values, and a `Usage` column.

### Dataset Details

| Property | Description |
|---|---|
| Total rows loaded | 6,626 |
| Image dimensions | 48 × 48 pixels |
| Image channels | 1 (grayscale) |
| Pixel values per image | 2,304 |
| Number of emotion classes | 7 |
| File format | CSV |

### Emotion Categories

| Label | Emotion |
|---|---|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Sad |
| 5 | Surprise |
| 6 | Neutral |

## Custom Dataset Class

A custom `FERDataset` class is implemented by inheriting from PyTorch's `Dataset` class.

### `__init__()`

Initializes the dataset using the supplied DataFrame and resets its index.

### `__len__()`

Returns the total number of samples in the dataset.

### `__getitem__()`

Retrieves an individual image and its emotion label. The image is converted into a tensor, reshaped into `1 × 48 × 48`, and normalized. The corresponding emotion label is converted into a PyTorch integer tensor.

This custom Dataset class makes the facial image data compatible with PyTorch's data-loading pipeline.

## DataLoader

PyTorch's `DataLoader` is used to load images in batches for training, validation, and testing.

| DataLoader | Batch Size | Shuffle |
|---|---:|---|
| `train_loader` | 64 | True |
| `val_loader` | 64 | False |
| `test_loader` | 64 | False |

The training DataLoader shuffles samples to vary their order during training. The validation and testing DataLoaders preserve their sample order.

## Dataset Splitting

The notebook uses `train_test_split()` from Scikit-learn to divide the cleaned dataset into training, validation, and testing subsets.

The split is performed in two stages:

- **Training split:** 80% of the dataset is allocated for training.
- **Temporary split:** The remaining 20% is divided equally into validation and testing subsets.

The split uses `random_state=42` for reproducibility and stratifies the samples according to their emotion labels.

### Dataset Sizes

| Subset | Percentage | Number of Samples |
|---|---:|---:|
| Training | 80% | 5,300 |
| Validation | 10% | 663 |
| Testing | 10% | 663 |
| **Total** | **100%** | **6,626** |

**Note:** Although the original FER-2013 dataset provides `Training`, `PublicTest`, and `PrivateTest` partitions, this notebook creates a new stratified random split and does not use the `Usage` column to assign those original partitions.

## CNN Model Architecture

The model is implemented using a custom PyTorch neural network class named `FERNet`.

## CNN Model Architecture

> ### Convolutional Feature Extraction
>
> The feature extraction section contains two convolutional blocks:
>
> - **First Convolutional Block:** 1 input channel, 32 output channels, a 3 × 3 kernel, ReLU activation, batch normalization, and max pooling.
> - **Second Convolutional Block:** 32 input channels, 64 output channels, a 3 × 3 kernel, ReLU activation, batch normalization, and max pooling.
>
> The max-pooling layers reduce the spatial dimensions from 48 × 48 to 24 × 24 and then to 12 × 12.
>
> ### Fully Connected Classifier
>
> After feature extraction, the output is flattened and passed through fully connected layers.
>
> - **Flattened Feature Vector:** `64 × 12 × 12`
> - **First Fully Connected Layer:** 128 neurons with ReLU activation and dropout of 0.4.
> - **Second Fully Connected Layer:** 64 neurons with ReLU activation and dropout of 0.4.
> - **Output Layer:** 7 neurons, one for each emotion category.
>
> The final layer produces class scores, and the predicted emotion is selected using the highest-scoring class.

### CNN Architecture Diagram

<img width="531" height="286" alt="Screenshot 2026-10-13 at 10 32 23 PM" src="https://github.com/user-attachments/assets/a345a1bd-b662-49fa-a3aa-3f02ab82b046" />


## Model Training

The model is trained using PyTorch for **30 epochs**.

During each epoch, the training loop:

- Loads batches of images and labels from `train_loader`.
- Moves the data to the selected computation device.
- Performs the forward pass through the CNN.
- Calculates the loss using the actual labels.
- Clears the previous gradients using `optimizer.zero_grad()`.
- Computes gradients using `loss.backward()`.
- Updates model parameters using `optimizer.step()`.
- Calculates and prints the average training loss for the epoch.

### Training Configuration

| Parameter | Value |
|---|---|
| Model class | `FERNet` |
| Framework | PyTorch |
| Epochs | 30 |
| Batch size | 64 |
| Loss function | `nn.CrossEntropyLoss()` |
| Optimizer | Stochastic Gradient Descent (SGD) |
| Learning rate | 0.01 |
| Weight decay | 0.0001 |
| Random seed | 42 |

### Training Loss

The training loss decreases substantially over the 30 epochs.

| Epoch | Training Loss |
|---:|---:|
| 1 | 1.68521 |
| 5 | 1.29937 |
| 10 | 0.68964 |
| 15 | 0.07647 |
| 20 | 0.01500 |
| 25 | 0.00946 |
| 30 | 0.00578 |

The decreasing loss indicates that the model is fitting the training samples increasingly well.

## Model Evaluation

The model is evaluated using the validation DataLoader. The evaluation code uses `model.eval()` to switch the network to evaluation mode and `torch.no_grad()` to disable gradient calculations.

For each batch, the model predicts the emotion class using the highest output score. These predictions are compared with the actual labels to calculate validation accuracy.

### Validation Results

| Metric | Result |
|---|---:|
| Validation samples | 663 |
| Correct predictions | 662 |
| Incorrect predictions | 1 |
| Validation accuracy | 99.85% |

The validation accuracy is calculated as:

\[
\text{Accuracy}=\frac{662}{663}\times100
\]

\[
\text{Accuracy}\approx99.85\%
\]


## Testing on Five Facial Images

The final notebook cell selects five images from `test_loader` and compares the model's predicted emotions with their actual labels.

The model is switched to evaluation mode, and predictions are generated without calculating gradients.
All five selected images were classified correctly in the recorded notebook output.



## Technology Stack

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Google Colab

## Conclusion

This project demonstrates the implementation of a Facial Emotion Recognition system using a CNN built with PyTorch.

It covers image preprocessing, custom Dataset and DataLoader implementation, stratified data splitting, convolutional feature extraction, model training, loss calculation, validation, and individual image prediction.

The model achieved **99.85% validation accuracy** on the recorded validation split and **100% accuracy on five selected test images**. These results describe the current notebook run; evaluation on a separate, representative dataset is necessary to establish generalization performance.

