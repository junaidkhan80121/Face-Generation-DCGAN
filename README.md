# DCGAN Face Generation

This project trains a Deep Convolutional GAN (DCGAN) in PyTorch to generate human face images from the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset.

The repository currently contains a single notebook:

- `Face_Generation.ipynb` - end-to-end workflow for downloading CelebA, preprocessing images, defining the generator and discriminator, training the model, and visualizing generated samples.

## What the Notebook Does

The notebook walks through a standard DCGAN pipeline:

1. Downloads the CelebA dataset.
2. Extracts images into `data_faces/`.
3. Resizes images to `64x64`.
4. Normalizes images to `[-1, 1]`.
5. Defines a generator and discriminator using convolutional layers.
6. Trains both networks with binary cross-entropy loss and Adam optimizers.
7. Saves generated image grids during training.

## Model Details

### Hyperparameters

- Image size: `64`
- Batch size: `128`
- Latent vector size: `100`
- Epochs: `5`
- Optimizer: `Adam`
- Learning rate: `0.0002`
- Betas: `(0.5, 0.999)`

### Generator

The generator uses `ConvTranspose2d -> BatchNorm2d -> ReLU` blocks to progressively upsample a `100`-dimensional latent vector into a `3 x 64 x 64` RGB image, followed by `Tanh` at the output layer.

### Discriminator

The discriminator uses `Conv2d -> BatchNorm2d -> LeakyReLU` blocks to downsample an input image and output a single probability through `Sigmoid`.

### Weight Initialization

The notebook initializes:

- Convolution weights from `N(0, 0.02)`
- Batch normalization weights from `N(1, 0.02)`
- Batch normalization bias to `0`

## Requirements

Install the Python packages used in the notebook:

```bash
pip install torch torchvision matplotlib numpy pillow
```

You will also need:

- Python 3
- Jupyter Notebook or Google Colab
- A CUDA-enabled GPU if you want faster training

## Dataset

The notebook downloads CelebA with:

```bash
wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip
```

It then extracts the archive so the project structure becomes:

```text
data_faces/
└── img_align_celeba/
    ├── 000001.jpg
    ├── 000002.jpg
    └── ...
```

Because `torchvision.datasets.ImageFolder` is used with `root='./data_faces'`, the `img_align_celeba` directory acts as the single class folder.

## How To Run

1. Open `Face_Generation.ipynb` in Jupyter or Colab.
2. Run the dataset download and extraction cells.
3. Run the import and preprocessing cells.
4. Run the generator and discriminator definition cells.
5. Run the training cell.
6. View saved samples and plotted outputs.

## Important Note For Local Runs

The notebook saves generated images to:

```python
sample_dir = '/content/sample_data/images'
```

That path is specific to Google Colab. If you run the notebook locally, change it to something inside this repository, for example:

```python
sample_dir = 'samples'
os.makedirs(sample_dir, exist_ok=True)
```

## Outputs

During training, the notebook:

- Tracks generator loss
- Tracks discriminator loss
- Stores generated image grids in memory
- Saves fake images periodically as PNG files

## Repository Goal

This notebook is a compact educational implementation of DCGAN for face generation, suitable for learning:

- GAN training dynamics
- Image preprocessing in PyTorch
- Generator and discriminator design
- Sampling synthetic faces from random noise

## Acknowledgment

The dataset download link used in the notebook comes from Udacity's deep learning resources, and the notebook is configured to run comfortably in Google Colab.
