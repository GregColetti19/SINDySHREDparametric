import matplotlib.pyplot as plt
import numpy as np
import torch

criterion = torch.nn.MSELoss()

def test_error(model, test_set):
    '''
    Function to calculate the error of the model on the given test_set.
    
    Inputs:
    - model: The trained model to be evaluated.
    - test_set: The dataset to test the model on.
    
    Outputs:
    - error_shred: The error of the model's predictions on the test set.
    - error_sindy: The error of the SINDy model's predictions on the test set.
    '''
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        model.eval()
        error_shred = criterion(model(test_set.data.X[:,:-1,:]), test_set.data.Y[:,-1,:])
        sindy_input = model.gru(test_set.data.X[:,:-1,:])
        sindy_output = model.sindy.simulate_next_explicit(sindy_input, test_set.params)
        sindy_target = model.gru(test_set.data.X[:,1:,:])
        error_sindy = criterion(sindy_output, sindy_target)
        return error_shred, error_sindy
        
def plot_sensors_comparison_shred(model, test_set, nt, which_param):
    '''
    Function to plot the comparison between real values and model predictions for selected sensors.
    
    Inputs:
    - model: The trained model to be evaluated.
    - test_set: The dataset to test the model on.
    - nt: Number of time steps.
    - which_param: The parameter index to be plotted.
    
    Outputs:
    - None (plots the comparison).
    '''
    real_values = test_set.data.Y[:, -1, :].cpu().numpy()
    predictions = model(test_set.data.X).cpu().detach().numpy()
    sensor_locations = np.random.randint(0, real_values.shape[1], 3)
    k = which_param
    plt.figure(figsize=(12, 6))
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(f'sensor {sensor_locations[i]}, param {k+1}-th')
        plt.plot(real_values[k*nt+1:(k+1)*nt, sensor_locations[i]], label=f'Real Values Sensor {i}')
        plt.plot(predictions[k*nt+1:(k+1)*nt, sensor_locations[i]], label=f'Predictions Sensor {i}')
        plt.legend(loc='best')

def plot_sensors_comparison_sindy(model, test_set, nt, which_param):
    '''
    Function to plot the comparison between real values and SINDy model predictions for hidden states.
    
    Inputs:
    - model: The trained model to be evaluated.
    - test_set: The dataset to test the model on.
    - nt: Number of time steps.
    - which_param: The parameter index to be plotted.
    
    Outputs:
    - None (plots the comparison).
    '''
    target = model.gru(test_set.data.X[:, 1:, :]).detach().cpu().numpy()
    sindy_input = model.gru(test_set.data.X[:, :-1, :])
    sindy_output = model.sindy.simulate_next_explicit(sindy_input, test_set.params)
    predictions = sindy_output.detach().cpu().numpy()

    k = which_param
    hidden_size = model.gru.hidden_size
    model.sindy.print_equations()
    for i in range(hidden_size):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, i+1)
        plt.plot(target[k*nt+1:(k+1)*nt, i], label=f'z{i+1} real ')
        plt.subplot(1, 3, i+1)
        plt.title(f'param = {k+1}-th, Real-Predictions')
        plt.plot(predictions[k*nt+1:(k+1)*nt, i], label=f'z{i+1} prediction')
        plt.legend(loc='best')
        plt.show()

def plot_shred_comparison_1D(model, test_set, instant, nx, nt, nf, which_param):
    '''
    Function to plot the comparison between real values and model predictions for 1D data.
    
    Inputs:
    - model: The trained model to be evaluated.
    - test_set: The dataset to test the model on.
    - instant: The specific time instant to be plotted.
    - nx: Number of spatial points in the x-dimension.
    - nt: Number of time steps.
    - nf: Number of features.
    - which_param: The parameter index to be plotted.
    
    Outputs:
    - None (plots the comparison).
    '''
    num_param_test = int(test_set.data.X.shape[0] / nt)

    real_values = test_set.data.Y[:, -1, :].cpu().numpy()
    predictions = model(test_set.data.X[:, :-1, :]).cpu().detach().numpy()

    spatial_dim = nx 

    real_values = real_values.reshape(num_param_test, nt, spatial_dim, nf)
    predictions = predictions.reshape(num_param_test, nt, spatial_dim, nf)

    for i in range(nf):
        plt.figure(figsize=(12, 6))
        plt.imshow(real_values[which_param, instant, :, i], cmap='jet')
        plt.title(f'Param {which_param+1}-th, Real Values for Feature {i+1}')
        plt.colorbar()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.imshow(predictions[which_param, instant, :, i], cmap='jet')
        plt.title(f'Param {which_param+1}-th, Predictions for Feature {i+1}')
        plt.colorbar()
        plt.show()

def plot_shred_comparison_2D(model, test_set, instant, nx, ny, nt, nf, which_param):
    '''
    Function to plot the comparison between real values and model predictions for 2D data.
    
    Inputs:
    - model: The trained model to be evaluated.
    - test_set: The dataset to test the model on.
    - instant: The specific time instant to be plotted.
    - nx: Number of spatial points in the x-dimension.
    - ny: Number of spatial points in the y-dimension.
    - nt: Number of time steps.
    - nf: Number of features.
    - which_param: The parameter index to be plotted.
    
    Outputs:
    - None (plots the comparison).
    '''
    num_param_test = int(test_set.data.X.shape[0] / nt)

    real_values = test_set.data.Y[:, -1, :].cpu().numpy()
    predictions = model(test_set.data.X[:, :-1, :]).cpu().detach().numpy()

    spatial_dim = nx * ny

    real_values = real_values.reshape(num_param_test, nt, spatial_dim, nf)
    predictions = predictions.reshape(num_param_test, nt, spatial_dim, nf)

    for i in range(nf):
        plt.figure(figsize=(6, 3))
        plt.imshow(real_values[which_param, instant, :, i].reshape(nx, ny), cmap='viridis')
        plt.title(f'Param {which_param+1}-th, Real Values for Feature {i+1}, instant {instant}')
        plt.colorbar()
        plt.show()

        plt.figure(figsize=(6, 3))
        plt.imshow(predictions[which_param, instant, :, i].reshape(nx, ny), cmap='viridis')
        plt.title(f'Param {which_param+1}-th, Predictions for Feature {i+1}, instant {instant}')
        plt.colorbar()
        plt.show()