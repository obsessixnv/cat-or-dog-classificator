This project demonstrates how to classify images of dogs and cats using a Convolutional Neural Network (CNN) implemented in PyTorch. The model can be trained on a dataset of dog and cat images and then used to classify new images or perform real-time predictions using a webcam

Dataset
The dataset should be structured as follows:



This structure represents the organization of your dataset:

- `data/` is the main directory.
- `train/` and `validation/` are subdirectories containing training and validation datasets, respectively.
- `dogs/` and `cats/` are subdirectories under `train/` and `validation/`, containing images of dogs and cats.


You can use the [Kaggle Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data) for this project.

Acknowledgements
- PyTorch
- OpenCV
- Kaggle Dogs vs. Cats dataset
