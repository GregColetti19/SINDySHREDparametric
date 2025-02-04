import torch
import torch.nn as nn
import itertools
import torch.autograd as autograd
from torch.utils.data import DataLoader
from copy import deepcopy

# GRU module definition
class GRUmodule(nn.Module):
    def __init__(self, num_sensors, hidden_size=3, hidden_layers=2):
        super(GRUmodule, self).__init__()

        # Single GRU layer
        self.gru = nn.GRU(input_size=num_sensors, hidden_size=hidden_size,
                                num_layers=hidden_layers, batch_first=True)
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.num_sensors = num_sensors

    def forward(self, x):
        batch_size, lags, n_states = x.size()

        # Initialize hidden state
        h_0 = torch.zeros((self.hidden_layers, batch_size, self.hidden_size), dtype=torch.float)

        # Move to GPU if necessary
        if next(self.parameters()).is_cuda:
            h_0 = h_0.cuda()

        # GRU forward pass for each parameter
        _, h_out = self.gru(x, h_0)

        return h_out[-1]

# Polynomial feature generation module
class PolynomialFeatures(nn.Module):
    def __init__(self, feature_names, param_names, max_degree=2, include_bias=False):
        super(PolynomialFeatures, self).__init__()
        self.feature_names = feature_names
        self.param_names = param_names
        self.max_degree = max_degree
        self.include_bias = include_bias

    def get_feature_names(self):
        names = []
        if self.include_bias:
            names.append("1")  # Constant term

        # Add all combinations of features (including interaction terms)
        for degree in range(1, self.max_degree + 1):
            for terms in itertools.combinations_with_replacement(self.feature_names, degree):
                names.append("*".join(terms))  # Generate feature names correctly

        # Add parameter interactions for each hidden variable
        if self.param_names is not None:
            for param_name in self.param_names:
                names.append(param_name)  # Parameter alone
                for feature_name in self.feature_names:
                    names.append(f"{param_name}*{feature_name}")  # Parameter * Feature
        
        return names

    def forward(self, x, params=None):
        """
        Forward pass for the SindyShredModel.
        This method generates polynomial feature combinations and optionally includes
        parameter interactions with the input features.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).
            params (torch.Tensor, optional): Parameter tensor of shape (batch_size, num_params) 
                                             or (num_params,). If None or empty, an empty tensor 
                                             is used. Default is None.
        Returns:
            torch.Tensor: Concatenated tensor of generated features and parameter interactions 
                          of shape (batch_size, num_generated_features).
        Notes:
            - If `self.include_bias` is True, a bias term (column of ones) is included in the features.
            - Polynomial feature combinations are generated up to `self.max_degree`.
            - If `self.param_names` is not None, parameter interactions with features are included.
        """
        if params is None or params.numel() == 0:
            params = torch.tensor([], device=x.device)

        batch_size, num_features = x.shape
        features = [torch.ones((batch_size, 1), device=x.device)] if self.include_bias else []

        # Add polynomial feature combinations
        for degree in range(1, self.max_degree + 1):
            for terms in itertools.combinations_with_replacement(range(num_features), degree):
                term = torch.ones((batch_size,), device=x.device)
                for idx in terms:
                    term *= x[:, idx]
                features.append(term.unsqueeze(1))  # Ensure correct shape
        
        # Add parameters and their interactions with features
        if self.param_names is not None:
            num_params = len(self.param_names)  # Correct number of parameters
            if len(params.shape) == 1:
                params = params.unsqueeze(1)

            # Add parameters alone
            for j in range(num_params):
                features.append(params[:, j].unsqueeze(1))
                
            # Add only valid parameter-feature interactions
            for j in range(num_params):
                for i in range(num_features):
                    interaction_term = params[:, j] * x[:, i]
                    features.append(interaction_term.unsqueeze(1))
                    
        return torch.cat(features, dim=1)

# SINDy module definition
class SINDyModule(nn.Module):
    def __init__(self, output_features, dt, max_degree=2, feature_names=None, param_names=None, threshold=1e-5, include_bias=False):
        super(SINDyModule, self).__init__()
        self.library = PolynomialFeatures(
            feature_names=feature_names,
            param_names=param_names,
            max_degree=max_degree,
            include_bias=include_bias
        )
        self.feature_dim = len(self.library.get_feature_names())
        self.coefficients = nn.Linear(self.feature_dim, output_features, bias=include_bias)
        torch.nn.init.uniform_(self.coefficients.weight, a=0.0, b=2.0)
        self.threshold = threshold
        self.dt = dt

    def forward(self, x, params):
        library = self.library(x, params)
        return self.coefficients(library)

    def apply_threshold(self):
        with torch.no_grad():  # This operation does not require gradient computation
            self.coefficients.weight.data = torch.where(
                torch.abs(self.coefficients.weight.data) < self.threshold,
                torch.zeros_like(self.coefficients.weight.data),
                self.coefficients.weight.data
            )

    def print_equations(self):
        feature_names = self.library.get_feature_names()
        coeffs = self.coefficients.weight.detach().cpu().numpy()
        equations = []

        for i in range(coeffs.shape[0]):
            terms = [f"{coeff:.2f}*{name}" for coeff, name in zip(coeffs[i], feature_names) if abs(coeff) > self.threshold]
            equation = " + ".join(terms)
            equations.append(f"z{i+1}' = {equation}")

        for eq in equations:
            print(eq)

    def simulate_next_explicit(self, x_prev, params, ministeps=10):
        x_next = x_prev.clone().detach()
        dt_k = self.dt / ministeps
        for _ in range(ministeps):
            dx = self.forward(x_next, params)
            x_next = x_next + dt_k * dx

        return x_next

    def simulate_next_implicit(self, x_prev, params, tol=1e-6, max_iter=100):
        def implicit_function(state):
            return state - x_prev - self.dt * self.forward(state, params)

        state = x_prev.clone().detach().requires_grad_(True)
        for _ in range(max_iter):
            f_val = implicit_function(state)
            if torch.norm(f_val) < tol:
                break
            jacobian = autograd.functional.jacobian(implicit_function, state)
            delta = torch.linalg.solve(jacobian, -f_val)
            state = state + delta

        return state.detach()

    def simulate(self, initial_state, params, t_span, method='explicit'):
        simulation = torch.zeros((len(t_span) + 1, initial_state.shape[1]))
        simulation[0] = initial_state
        for t in range(len(t_span)):
            if method == 'explicit':
                simulation[t + 1] = self.simulate_next_explicit(simulation[t].unsqueeze(0), params).squeeze(0)
            elif method == 'implicit':
                simulation[t + 1] = self.simulate_next_implicit(simulation[t].unsqueeze(0), params).squeeze(0)
            else:
                raise ValueError("Method must be 'explicit' or 'implicit'")

        return simulation

# Decoder module definition
class DecoderModule(torch.nn.Module):
    def __init__(self, input_size, output_size, num_param, decoder_sizes=[10, 100], dropout=0.1):
        super(DecoderModule, self).__init__()

        self.decoder = torch.nn.ModuleList()
        self.num_param = num_param
        decoder_sizes.insert(0, input_size)
        decoder_sizes.append(output_size)

        for i in range(len(decoder_sizes)-1):
            self.decoder.append(torch.nn.Linear(decoder_sizes[i], decoder_sizes[i+1]))
            if i != len(decoder_sizes)-2:
                self.decoder.append(torch.nn.Dropout(dropout))
                self.decoder.append(torch.nn.ReLU())

    def forward(self, x):
        output = x
        for layer in self.decoder:
            output = layer(output)

        return output

# SINDySHRED model definition
class SINDySHRED(torch.nn.Module):
    def __init__(self, max_degree: int, features_name: list, dt: float, param_names: list, threshold: float,
                 num_sensors: int, num_param: int, n_states: int, hidden_size: int, hidden_layers: int,
                 decoder_sizes=[50, 100], dropout=0.1):
        super(SINDySHRED, self).__init__()

        self.gru = GRUmodule(num_sensors=num_sensors, hidden_size=hidden_size, 
                             hidden_layers=hidden_layers)
        self.sindy = SINDyModule(output_features=hidden_size, max_degree=max_degree, dt=dt,
                                 feature_names=features_name, param_names=param_names,
                                 threshold=threshold)
        self.decoder = DecoderModule(input_size=hidden_size, output_size=n_states, num_param=num_param,
                                     decoder_sizes=decoder_sizes, dropout=dropout)

    def forward(self, x):
        z = self.gru(x)
        x_approx = self.decoder(z)

        return x_approx

# Function to fit the SINDySHRED model
def fit_SindyShred_param(model, train_dataset, valid_dataset, batch_size, num_epochs, lr, lambda_sindy, verbose=True, patience=5):
    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    train_error_list = []
    valid_error_list = []
    patience_counter = 0
    best_params = model.state_dict()

    for epoch in range(1, num_epochs + 1):
        for data, params in train_loader:
            model.train()
            optimizer.zero_grad()
            shred_output = model(data[0][:,1:,:])
            loss_shred = criterion(shred_output, data[1][:,-1,:])
            sindy_input = model.gru(data[0][:,:-1,:])
            sindy_target = model.gru(data[0][:,1:,:])
            sindy_output = model.sindy.simulate_next_explicit(sindy_input, params)
            loss_sindy = criterion(sindy_output, sindy_target)
            loss_sparsity = lambda_sindy*torch.norm(model.sindy.coefficients.weight, 0)
            loss = loss_shred + loss_sindy + loss_sparsity
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            # Train loss
            loss_shred = criterion(model(train_dataset.data.X[:,:-1,:]), train_dataset.data.Y[:,-1,:])
            sindy_input = model.gru(train_dataset.data.X[:,:-1,:])
            sindy_target = model.gru(train_dataset.data.X[:,1:,:])
            sindy_output = model.sindy.simulate_next_explicit(sindy_input, train_dataset.params)
            loss_sindy = criterion(sindy_output, sindy_target)
            train_error = loss_shred + loss_sindy
            train_error_list.append(train_error)
            # Validation loss
            loss_shred = criterion(model(valid_dataset.data.X[:,:-1,:]), valid_dataset.data.Y[:,-1,:])
            sindy_input = model.gru(valid_dataset.data.X[:,:-1,:])
            sindy_output = model.sindy.simulate_next_explicit(sindy_input, valid_dataset.params)
            sindy_target = model.gru(valid_dataset.data.X[:,1:,:])
            loss_sindy = criterion(sindy_input, sindy_target)
            valid_error = loss_shred + loss_sindy
            valid_error_list.append(valid_error)

        if verbose:
            print(f"Training epoch {epoch}")
            print(f"Training error: {train_error}")
            print(f"Validation error: {valid_error}")

        if epoch % 10 == 0:
            model.sindy.apply_threshold()

        if verbose and (epoch == 1 or epoch % 10 == 0):
            model.sindy.print_equations()

        # Save parameters
        if valid_error == torch.min(torch.tensor(valid_error_list)):
            patience_counter = 0
            best_params = deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter == patience:
            model.load_state_dict(best_params)
            print("Patience reached")
            return torch.tensor(valid_error_list).cpu(), torch.tensor(train_error_list).cpu()

    model.load_state_dict(best_params)
    return torch.tensor(valid_error_list).cpu(), torch.tensor(train_error_list).cpu()
