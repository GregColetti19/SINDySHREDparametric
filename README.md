# parametric SINDy-SHRED: Combining Discovery and Prediction for Complex PDE Dynamics

## Overview
Parametric SINDy-SHRED is a hybrid machine learning framework that integrates Sparse Identification of Nonlinear Dynamics (SINDy) with a Shallow Recurrent Decoder (SHRED). This approach enables accurate discovery and prediction of complex partial differential equation (PDE) dynamics using a minimal set of sensor measurements.

The main contributions of this repository include:
- Reconstruction of the full spatial domain from sparse sensor data.
- Symbolic representation of the latent state via SINDy.
- Integration of parametric dependencies for improved generalization.
- Training and testing routines to evaluate model performance.

## Installation
To use this repository, first clone it and install the required dependencies:

```bash
git clone https://github.com/GregColetti19/SINDySHREDparametric.git
cd SINDySHREDparametric
```

## Usage
A tutorial is provided to guide you through the entire pipeline. Open the Jupyter Notebook:

```bash
jupyter notebook TutorialFlow.ipynb
```

The notebook will walk you through:
1. **Loading and reshaping data**
2. **Preprocessing data** (ordering, padding, lagging, splitting)
3. **Defining the model** with customizable parameters
4. **Training the model** using SINDy-SHRED
5. **Testing and visualizing results**

### Running the Model Manually
If you wish to run the model outside the notebook, use the following steps in Python:

#### **1. Load and Preprocess Data**
```python
from DataPreProcessing import processData

data, dt = load_your_data() #load your data and the temporal step
params = load_your_params() #load your params
data = reshape_to_correct_dimensions(data) #(nx, ny, nt, nf, num_param)
params = reshape_params(params) #(num_param, num_paramsForSystem)
train_dataset, val_dataset, test_dataset = processData(data, parameters=params, lags=11, num_sensors=40, train_ratio=0.8, val_ratio=0.1, param_split = True, constant_params=True, noise=None)
```

#### **2. Define the Model**
```python
from SindyShredModel import SINDySHRED

model = SINDySHRED(max_degree=1, hidden_size=3, hidden_layers=2, threshold=0.3,
                    num_sensors=40, decoder_sizes=[10, 500], dropout=0.1)
```

#### **3. Train the Model**
```python
from SindyShredModel import fit_SindyShred_param

train_error, val_error = fit_SindyShred_param(model, train_dataset, val_dataset,
                                              batch_size=25, num_epochs=1000,
                                              lr=0.01, lambda_sindy=0.3, verbose=True, patience=5)
```

#### **4. Test and Plot Results**
```python
from TestSindyShred import test_error, plot_sensors_comparison_shred, plot_sensors_comparison_sindy, plot_shred_comparison_2D

test_err_shred, test_err_sindy = test_error(model, test_dataset)
k = np.random.randint(0, int(num_param*0.1))
plot_sensors_comparison_shred(model, data_test, nt, k)
plot_sensors_comparison_sindy(model, data_test, nt, k)
instant = np.random.randint(0,nt)
plot_shred_comparison_2D(model, data_test, instant, nx, ny, nt, nf, k)
```

## File Structure
```
SINDySHREDparametric/
│-- DataPreProcessing.py  # Data preparation utilities
│-- SindyShredModel.py    # Model definition (GRU, SINDy, Decoder)
│-- TestSindyShred.py     # Testing and visualization utilities
│-- TutorialFlow.ipynb    # Step-by-step Jupyter Notebook tutorial
│-- requirements.txt      # Python dependencies
│-- README.md             # Documentation
```

## Citation
If you use this repository in your research, please cite:
```
@article{coletti2024sindyshred,
  author = {Gregorio Coletti},
  title = {Parametric SINDy-SHRED: Combining Discovery and Prediction for Complex PDE Dynamics},
  year = {2024},
  journal = {arXiv preprint}
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions, please reach out to **Gregorio Coletti** at gregoriocoletti@yahoo.com.

