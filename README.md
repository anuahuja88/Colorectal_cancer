# Deep Learning Classification of Microsatellite Status in Colorectal Cancer Whole Slide Images

## Project Overview

This project focuses on classifying the microsatellite status (MSS or MSI) in colorectal cancer whole slide images using deep learning techniques. The classification process involves several stages, including feature enhancement, normalization, and training a ResNet-18 model.

### Repository Structure

- `features.py`: This script is used to enhance certain features in MSI and MSS tiles. Techniques such as Contrast Limited Adaptive Histogram Equalization (CLAHE) and Local Binary Pattern (LBP) are implemented in this file.
- `resNetModel.py`: This script contains the code for training a ResNet-18 model. It takes in data tiles of MSI and MSS and trains the model to classify them accurately.
- `h&e_normalise.py`: This script handles the splitting of H&E stained tiles into their separate Hematoxylin (H) and Eosin (E) components for further analysis and processing.

### Project Goal

The primary goal of this project is to accurately classify colorectal cancer whole slide images as either Microsatellite Stable (MSS) or Microsatellite Instable (MSI) using a deep learning approach. This classification aids in better understanding the tumor biology and tailoring appropriate treatment strategies for patients.

## Getting Started

### Prerequisites

- Python 3.x
- Required Python libraries: numpy, pandas, scikit-learn, tensorflow, keras, opencv-python

### Installation

Clone the repository:
    ```
    git clone https://github.com/yourusername/colorectal-cancer-classification.git
    ```


### Usage

#### Feature Enhancement

To enhance the features of MSI and MSS tiles, run the `features.py` script:
```bash
python features.py 
```
This will apply techniques like CLAHE and LBP to the image tiles to enhance their distinguishing features.

#### H&E Normalization

To split the H&E stained tiles into their H and E components, run the `h&e_normalise.py` script:
```bash
python h&e_normalise.py
```
This will process the tiles and separate the Hematoxylin and Eosin components for further analysis.

#### Model Training

To train the ResNet-18 model on the data tiles, run the `resnetmodel.py` script:
```bash
python resNetModel.py
```
This will start the training process using the enhanced features and normalized H&E components. The trained model will be saved for later inference and evaluation.


## Contributors

### Anu Ahuja
**Title:** Researcher  
**Email:** aah109@uclive.ac.nz

### Ramakrishnan Mukundan
**Title:** Supervisor  
**Email:** mukundan@canterbury.ac.nz

### Arthru Morley-Bunker
**Title:** University of Otago Pathology and Biomedical Science Researcher
**Email:** arthur.morley-bunker@otago.ac.nz