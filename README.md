# Custom-Resnet50-with-Adamw-on-CIFAR-10
Description: Developed a convolutional neural network from scratch using PyTorch to classify CIFAR-10 images into 10 categories. Implemented a custom ResNet50 architecture with residual blocks and achieving 90.89% test accuracy. 
Technologies used: PyTorch, Numpy, Pandas, Matplotlib.
Link: https://www.kaggle.com/code/vmkhoa28/custom-resnet50-with-adamw

# Image Classification with ResNet50

## Purpose
This is my first project applying deep CNN architectures to an image classification task using the CIFAR-10 dataset.

## Objective
To classify CIFAR-10 images into one of the 10 categories using a custom implementation of the ResNet50 architecture.

## Dataset
- **Dataset**: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Size**: 60,000 color images (32x32), 10 classes
  - 50,000 for training
  - 10,000 for testing
- **Classes**: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

## Preprocessing
- Data Augmentation (for training/dev):
  - Random Horizontal Flip
  - Random Rotation (±20°)
  - Random Crop (32x32 with padding = 4)
- Normalization using ImageNet mean & std:
  - `mean = [0.485, 0.456, 0.406]`
  - `std = [0.229, 0.224, 0.225]`

## Model Architecture
- Custom ResNet50 built from scratch using PyTorch
- Residual Block:
  - 3 Convolutional layers: 1x1 → 3x3 → 1x1
  - BatchNorm after each Conv
  - Skip connection with optional identity downsampling
- Full model:
  - Input Conv + BN + ReLU
  - 4 layers with block repetitions: `[3, 4, 6, 3]`
  - Adaptive Average Pooling
  - Fully Connected Layer: `512 * 4 → 10`
- Activation: ReLU
- No max pooling (Identity used)

## Training
- **Epochs**: 50  
- **Loss Function**: `nn.CrossEntropyLoss()`  
- **Optimizer**: `torch.optim.AdamW`  
  - Learning rate: `0.001`  
  - Weight decay: `0.01`  
- **Batch Size**: 256  
- **Device**: CUDA (if available)  
- **Early Stopping**: Patience = 10  
- Metrics logged:
  - Training & validation accuracy
  - Training & validation loss

## Evaluation
- **Final Test Accuracy**: **90.92%**
- **Per-Class Accuracy**:
  - Airplane: 93.10%
  - Automobile: 95.30%
  - Bird: 84.80%
  - Cat: 84.10%
  - Deer: 90.30%
  - Dog: 84.00%
  - Frog: 98.20%
  - Horse: 92.20%
  - Ship: 94.10%
  - Truck: 93.10%

## Visualization
- Accuracy and loss curves for both train and validation sets plotted per epoch.
- Image samples shown with original class labels from the CIFAR-10 training set.

## Result Highlights
- Achieved **>90% test accuracy**
- Effective training using data augmentation and AdamW optimizer
- Balanced generalization (no severe overfitting)
