# Custom-Resnet50-with-Adamw-on-CIFAR-10
https://www.kaggle.com/code/vmkhoa28/custom-resnet50-with-adamw-2

## Purpose
This is my first project applying deep CNN architectures to an image classification task using the CIFAR-10 dataset.

## Objective
To classify CIFAR-10 images into one of the 10 categories using a custom ResNet50 architecture.

## Dataset
- **Dataset**: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Size**: 60,000 color images (32x32), 10 classes
  - 50,000 for training
  - 10,000 for testing
- **Classes**: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

## Preprocessing
- Data Augmentation for train and validation sets:
  - Random Horizontal Flip
  - Random Rotation
  - Random Crop (32x32 with padding = 4)
- Normalization using ImageNet mean & std:
  - `mean = [0.485, 0.456, 0.406]`
  - `std = [0.229, 0.224, 0.225]`

## Model Architecture
- Custom ResNet50 built from scratch using PyTorch
- Residual Block:
  - 3 Convolutional layers: 1x1 → 3x3 → 1x1
  - BatchNorm after each Conv
  - Skip connection with identity downsampling
- Full model:
  - Input Conv + BN + ReLU
  - 4 layers: `[3, 4, 6, 3]`
  - Adaptive average pooling
  - Fully connected layer: `512 * 4 → 10`
- Activation: ReLU
- No max pooling (The size is too small)

## Training
- **Epochs**: 50  
- **Loss Function**: `nn.CrossEntropyLoss()`  
- **Optimizer**: `torch.optim.AdamW`  
  - Learning rate: `0.001`  
  - Weight decay: `0.01`  
- **Batch Size**: 256  
- **Device**: CUDA (if available)  
- **Early Stopping**: Patience = 10  

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

