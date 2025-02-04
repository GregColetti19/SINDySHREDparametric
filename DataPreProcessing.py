import numpy as np
import torch
from torch.utils.data import Dataset

def dataOrdering(data):
    '''
    This function is used to order the data in the form (num_param, nt, n_states)
    make sure that the input data is of the form (nx, nt, nf, num_param) or (nx, ny, nt, nf, num_param) or (nx, ny, nz, nt, nf, num_param).
    where: 
    -nx, ny, nz are the spatial dimensions, 
    -nt is the number of time steps, 
    -nf is the number of features 
    -num_param is the number of parameters

    Parameters:
    - data: Input data of shape (nx, nt, nf, num_param) or (nx, ny, nt, nf, num_param) or (nx, ny, nz, nt, nf, num_param)

    Returns:
    - A NumPy array of shape (num_param, nt, n_states)
    '''
    if len(data.shape) == 4:
        nx, nt, nf, num_param = data.shape
        spatial_dim = nx 
        data_t = data.transpose(3, 1, 0, 2) # num_param, nt, nx, nf
    elif len(data.shape) == 5:
        nx, ny, nt, nf, num_param = data.shape
        spatial_dim = nx * ny
        data_t = data.transpose(4, 2, 0, 1, 3) # num_param, nt, nx, ny, nf
    elif len(data.shape) == 6:
        nx, ny, nz, nt, nf, num_param = data.shape
        spatial_dim = nx * ny * nz 
        data_t = data.transpose(5, 3, 0, 1, 2, 4) # num_param, nt, nx, ny, nz, nf
    else:
        Warning('Data shape is not supported, it should of the form (nx, nt, nf, num_param) or (nx, ny, nt, nf, num_param) or (nx, ny, nz, nt, nf, num_param)')
    
    n_states = spatial_dim * nf
    print("n_states: ", n_states)
    
    return data_t.reshape(num_param, nt, n_states)

def padding(data, pad_size):
    '''
    This function is used to pad the data with pad_size zeros, along the temporal dimension.

    Parameters:
    - data: Input data of shape (num_param, nt, n_states)
    - pad_size: Number of zeros to pad along the temporal dimension

    Returns:
    - Padded data of shape (num_param, nt + pad_size, n_states)
    '''
    num_param, nt, n_states = data.shape
    data_padded = np.zeros((num_param, nt + pad_size, n_states))
    data_padded[:, pad_size:, :] = data
    return data_padded

def lagging(data, lags):
    '''
    This function is used to lag the data by lags time steps.

    Parameters:
    - data: Input data of shape (num_param, lagged_time, n_states)
    - lags: Number of time steps to lag the data

    Returns:
    - Lagged data of shape (num_param * nt, lags, n_states)
    '''
    num_param, lagged_time, n_states = data.shape
    nt = lagged_time - lags
    data_lag = np.zeros((num_param * nt, lags, n_states))

    for i in range(num_param):
        for j in range(nt):
            data_lag[i * nt + j] = data[i, j:j + lags, :]

    return data_lag

class SpiralDataset_2inputs(Dataset):
    def __init__(self, data, params):
        """
        Initialize the dataset.

        Parameters:
        - data: Tensor of input data (e.g., features over time)
        - params: Tensor of beta values corresponding to time steps
        """
        self.data = data
        self.params = params

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get the data and parameters at the specified index.

        Parameters:
        - idx: Index of the data to retrieve

        Returns:
        - Tuple of (data, params) at the specified index
        """
        return self.data[idx], self.params[idx]
    
def addNoise(data, noise):
    '''
    This function adds Gaussian noise to the data.

    Parameters:
    - data: Input data of shape (nt*num_param, lags, n_states)
    - noise: Standard deviation of the Gaussian noise to be added

    Returns:
    - Noisy data of the same shape as input data
    '''
    noisy_data = data + np.random.normal(0, noise, data.shape)
    return noisy_data

class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if idx >= len(self.X):
            raise IndexError(f"Requested index {idx} exceeds dataset length {len(self.X)}")
        return self.X[idx], self.Y[idx]

def dataSplitting(data, parameters, nt, num_param, num_sensors, train_ratio=0.8, val_ratio=0.1, param_split = True, constant_params=True, noise=None):
    '''
    This function is used to split the data into training, validation and test sets.

    Parameters:
    - data: Input data of shape (nt*num_param, lags, n_states)
    - parameters: Array of parameter values
    - nt: Number of time steps
    - num_param: Number of parameters
    - num_sensors: Number of sensors to select (default: 32)
    - train_ratio: Ratio of training data (default: 0.8)
    - val_ratio: Ratio of validation data (default: 0.1)
    - constant_params: Whether to use constant parameters (default: True)
    - noise: Standard deviation of the Gaussian noise to be added (default: None)

    Returns:
    - Tuple of (train_dataset, valid_dataset, test_dataset)
    '''
    ntxnum_param, lags, n_states = data.shape
    sensor_locations = np.random.choice(n_states, num_sensors, replace=False)
    lagged_data_sens = data[:, :, sensor_locations]
    if noise is not None:
        lagged_data_sens = addNoise(lagged_data_sens, noise)

    print("lagged_data_sens shape: ", lagged_data_sens.shape)

    
    
    if param_split:
        # Generate indices for each parameter
        param_indices = np.arange(num_param)

        # Shuffle the parameter indices
        np.random.shuffle(param_indices)
        # Split the parameter indices
        train_param_indices = param_indices[:int(num_param * train_ratio)]
        valid_param_indices = param_indices[int(num_param * train_ratio):int(num_param * (train_ratio + val_ratio))]
        test_param_indices = param_indices[int(num_param * (train_ratio + val_ratio)):]

        # Generate the actual indices for training, validation, and testing
        train_indices = np.concatenate([np.arange(k * nt, (k + 1) * nt) for k in train_param_indices])
        valid_indices = np.concatenate([np.arange(k * nt, (k + 1) * nt) for k in valid_param_indices])
        test_indices = np.concatenate([np.arange(k * nt, (k + 1) * nt) for k in test_param_indices])
    else:
        # Split the data along the time dimension for each parameter
        train_indices = np.concatenate([np.arange(k * nt, k * nt + int(nt * train_ratio)) for k in range(num_param)])
        valid_indices = np.concatenate([np.arange(k * nt + int(nt * train_ratio), k * nt + int(nt * (train_ratio + val_ratio))) for k in range(num_param)])
        test_indices = np.concatenate([np.arange(k * nt + int(nt * (train_ratio + val_ratio)), (k + 1) * nt) for k in range(num_param)])

    # Organize data for training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if constant_params:
        params = torch.tensor(parameters, dtype=torch.float32).repeat_interleave(nt).unsqueeze(1).to(device)
    else:
        params = torch.tensor(parameters, dtype=torch.float32).to(device)

    # Parameters
    param_basic = params[train_indices]
    param_valid = params[valid_indices]
    param_test = params[valid_indices]
    # Dataset
    train_basic_in = torch.tensor(lagged_data_sens[train_indices,:,:], dtype=torch.float32).to(device)
    valid_basic_in = torch.tensor(lagged_data_sens[valid_indices,:,:], dtype=torch.float32).to(device)
    test_basic_in = torch.tensor(lagged_data_sens[test_indices,:,:], dtype=torch.float32).to(device)

    train_basic_out = torch.tensor(data[train_indices,:,:], dtype=torch.float32).to(device)
    valid_basic_out = torch.tensor(data[valid_indices,:,:], dtype=torch.float32).to(device)
    test_basic_out = torch.tensor(data[test_indices,:,:], dtype=torch.float32).to(device)

    # CONNECT in-and-out
    # BASIC
    train_basic_set = TimeSeriesDataset(train_basic_in, train_basic_out)
    valid_basic_set = TimeSeriesDataset(valid_basic_in, valid_basic_out)
    test_basic_set = TimeSeriesDataset(test_basic_in, test_basic_out)

    train_dataset = SpiralDataset_2inputs(train_basic_set, param_basic)
    valid_dataset = SpiralDataset_2inputs(valid_basic_set, param_valid)
    test_dataset = SpiralDataset_2inputs(test_basic_set, param_test)

    return train_dataset, valid_dataset, test_dataset

def processData(data, parameters, lags, num_sensors, train_ratio=0.8, val_ratio=0.1, param_split = True, constant_params=True, noise=None):
    '''
    This function processes the input data by ordering, padding, lagging, and splitting it into training, validation, and test sets.

    Parameters:
    - data: Input data of shape (nx, nt, nf, num_param) or (nx, ny, nt, nf, num_param) or (nx, ny, nz, nt, nf, num_param)
    - parameters: Array of parameter values
    - pad_size: Number of zeros to pad along the temporal dimension
    - lags: Number of time steps to lag the data
    - num_sensors: Number of sensors to select (default: 32)
    - train_ratio: Ratio of training data (default: 0.8)
    - val_ratio: Ratio of validation data (default: 0.1)
    - constant_params: Whether to use constant parameters (default: True)
    - noise: Standard deviation of the Gaussian noise to be added (default: None)

    Returns:
    - Tuple of (train_dataset, valid_dataset, test_dataset)
    '''
    # Order the data
    ordered_data = dataOrdering(data)
    print("data ordered")
    
    # Pad the data
    padded_data = padding(ordered_data, lags)
    print("data padded")
    
    # Lag the data
    lagged_data = lagging(padded_data, lags)
    print("data lagged")

    
    # Split the data
    train_dataset, valid_dataset, test_dataset = dataSplitting(lagged_data, parameters, ordered_data.shape[1], ordered_data.shape[0], num_sensors, train_ratio, val_ratio, param_split, constant_params, noise)
    print("data split")
    
    return train_dataset, valid_dataset, test_dataset
