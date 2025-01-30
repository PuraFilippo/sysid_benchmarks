
import pickle
from transformer_sim import Config, TSTransformer
from pathlib import Path
from trajGenerator import trajGenerator
import datetime
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import config as cfg  
from nonlinear_benchmarks.error_metrics import RMSE, NRMSE, R_squared, MAE, fit_index
from dynonet.lti import MimoLinearDynamicalOperator
from dynonet.static import MimoStaticNonLinearity
import os
import copy

default_benchmark = 'CED'

def parse_args():
    """
    Parses command-line arguments to configure the model and training settings.
    """

    parser = argparse.ArgumentParser(description='dynoNet training on various datasets')
    parser.add_argument('--benchmark_name', type=str, default=default_benchmark, help='Type of model to train')
    
    # Temporary args parsing to get model type
    temp_args, _ = parser.parse_known_args()
    config = cfg.get_config(temp_args.benchmark_name)
    
    # Add arguments dynamically based on current model configuration
    for key, val in config.items():
        if key != 'command_load' and not any(action.dest == key for action in parser._actions):  # Prevent duplication
            parser.add_argument(f'--{key}', type=type(val), default=val, help=f'Set {key} (default: {val})')

    args = parser.parse_args()
    return args, config

def get_config():
    """
    Loads configuration using command-line arguments and updates global configuration.
    """
    args, config = parse_args()

    # Update the model configuration with command line arguments
    for key in config:
        if hasattr(args, key):
            config[key] = getattr(args, key)

    print("Configuration used:", config)

    return config



def set_seed(np_seed=42, torch_seed = 42):
    
    """Set seed for reproducibility."""

    np.random.seed(np_seed)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SubSeqDataset(Dataset):
    """
    Custom dataset to handle sub-sequences of input-output pairs for training.
    """
        
    def __init__(self, train, seq_len):
        # Initialize lists to store sequences
        self.u_list = []
        self.y_list = []

        # Process each (u_train, y_train) pair in the training data
        for u_train, y_train in train:
            N, nu = u_train.shape  # Get the total number of samples and features in u
            _, ny = y_train.shape  # Get the total number of features in y

            # Calculate the number of full sequences possible, minus the last potentially shorter sequence
            n = int(np.ceil(N / seq_len))

            # Iterate through each sequence index
            for ind in range(n):
                start_idx = ind * seq_len
                end_idx = start_idx + seq_len

                # Adjust the start index of the last sequence if it's shorter than seq_len
                if end_idx > N:
                    start_idx = max(0, N - seq_len)  # Start the last sequence so that it ends exactly at the last data point
                    end_idx = N

                # Extract the sequences
                u_subseq = u_train[start_idx:end_idx, :]
                y_subseq = y_train[start_idx:end_idx, :]

                # Store the sequences
                self.u_list.append(u_subseq)
                self.y_list.append(y_subseq)

        # Convert lists to numpy arrays and then to PyTorch tensors
        self.u = torch.from_numpy(np.array(self.u_list, dtype=np.float32))
        self.y = torch.from_numpy(np.array(self.y_list, dtype=np.float32))

    def __len__(self):
        return len(self.u_list)

    def __getitem__(self, idx):
        return self.u[idx], self.y[idx]

class Scaler:
    """
    Scaler class for normalizing and denormalizing data.
    """

    def __init__(self):
        self.mean = None
        self.std = None
        

    def fit(self, data):
        """Fit the scaler to the data by calculating mean and standard deviation."""
        self.mean = data.mean()
        self.std = data.std()

    def transform(self, data):
        """Normalize the data using the calculated mean and standard deviation."""
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """Revert the normalization to return to the original scale."""
        return data * self.std + self.mean

class my_dynoNet(nn.Module):
    '''
    dynoNet model definition
    '''

    def __init__(self, input_size, hidden_size, output_size, n_a, n_b):
        super(my_dynoNet, self).__init__()

        
        self.G1 = MimoLinearDynamicalOperator(
            in_channels=input_size, out_channels=1, n_b=n_b, n_a=n_a, n_k = 1
        )
        self.F = MimoStaticNonLinearity(1, 1, n_hidden=hidden_size)
        self.G2 = MimoLinearDynamicalOperator(
            in_channels=1, out_channels=output_size, n_b=n_b, n_a=n_a
        )
        self.Glin = MimoLinearDynamicalOperator(
            in_channels=input_size, out_channels=output_size, n_b=n_b+1, n_a=n_a
        )

    def forward(self, x_in):

        x = self.G1(x_in)
        x = self.F(x)
        x = self.G2(x)
        y = x + self.Glin(x_in)

        return y

def train_dynoNet(train, test, in_context_model,max_epochs,print_frequency, sigmay_train = 1,gamma_list = [1], regularization = True):
    """
    Trains the DynoNet model with provided datasets and data loader for generating synthetic data.

    Args:
    train, test(list): the training dataset, and testing dataset
    in_context_model: The trained in-context learner trained on WH systems 
    max_epochs: Number of training epochs.
    print_frequency: Frequency of printing the training progress.
    sigmay_train: Standard deviation on training data.
    gamma_list: List of regularization parameters. Default: [1].
    regularization: Boolean indicating if regularization is applied. If not, variance on the estimation error is used to weight synthetic data

    Returns:
    train_loss_dict, val_loss_dict, ts_loss_dict: Dictionaries containing training, validation and test losses mean over the number of trajectories.
    test_loss_dict: Dictionary containing the test loss for each trajectory
    val_r2_dict: Dictionary containing the mean R2 index for the validation part
    test_r2_dict: Dictionary containing the R2 index for the test part for both trajectories
    """

    # Unpacking training and validation data
    lr = config['lr']
    n_skip = config['n_skip']
    # Context length
    seq_len = config['seq_len']
    batch_size = config['batch_size']
    shuffle = config['shuffle']

    # normalization of the training data
    train_2, u_scaler, y_scaler = data_normalizer(train)
    my_dataset = SubSeqDataset(train_2, seq_len)
    loader = DataLoader(my_dataset, shuffle=shuffle, batch_size=batch_size)
    u, y = next(iter(loader))
    u,y = u.to(device, non_blocking=True),y.to(device, non_blocking=True)

    # lenght of the synthetic sequence 
    len_sim = 200

    # generator of synthetic data from the trasformer
    tG_ds = trajGenerator(model=in_context_model, u_ctx=u[:, :400, :], y_ctx=y[:, :400, :], device = device, len_sim=len_sim)
    tG_dl = DataLoader(tG_ds, batch_size=1, num_workers=0)

    # Loss tracking
    train_loss_dict, val_loss_dict, test_loss_dict, ts_loss_dict  = {}, {}, {}, {}
    val_r2_dict, test_r2_dict = {}, {}

    # losses over the gamma values
    total_losses = []
    total_v_losses = []
    total_ts_losses = []
    # total_v_losses_2 = [] #for the validation loss for the second trajectory
    total_test_losses = []
    # total_test_losses_2 = [] #for the test loss for the second trajectory
    idx_best = []

    # Creating DynoNet model components
    model_original = my_dynoNet(input_size=1, hidden_size=20, output_size=1, n_a=5, n_b=5)

    
    for gamma in gamma_list:
        
        model = copy.deepcopy(model_original)
        # Setting up the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)


        val_loss_best = np.inf
        # val_loss_best_2 = np.inf
        train_loss_best = np.inf
        # test_loss_best_2 = np.inf
        val_r2_best = 0
        test_r2_best = 0
        
        #losses for all the epochs in the training
        total_losses_row = []
        total_v_losses_row = []
        total_ts_losses_row = []
        # total_v_losses_2_row = [] #for the validation loss for the second trajectory
        total_test_losses_row = []
        # total_test_losses_2_row = [] #for the test loss for the second trajectory
        total_loss_sim = []
        total_loss_ctx = []

        for epoch in range(max_epochs):

            # Reset gradients
            optimizer.zero_grad()

            # generate the input/output from the dataloader
            for u, y in loader:
                
                ctx_len = 300
                val_len = 100
                u,y = u.to(device, non_blocking=True),y.to(device, non_blocking=True)

                u_t, y_t = u[:, :ctx_len, :], y[:, :ctx_len, :] 
                u_t, y_t = u_t.cpu(), y_t.cpu()  # Training data

                # Generate synthetic data and concatenate with context data
                with torch.no_grad():
                    # generate synthetic data
                    u_sim, y_sim, sigmay_sim = next(iter(tG_dl))
                    u_sim, y_sim, sigmay_sim = u_sim.squeeze(0), y_sim.squeeze(0), sigmay_sim.squeeze(0)
                    u_sim = u_sim[:,:,:].view(2,len_sim,nu).cpu()
                    y_sim = y_sim[:,:,:].view(2,len_sim,nu).cpu()
                    sigmay_sim = sigmay_sim.cpu()
                    # concatenation of the input data
                    u_conc = torch.concat((u_t, u_sim), dim=1)


                # # Forward pass: Simulate the DynoNet
                y_hat = model(u_conc)


                # # Compute loss for context and synthetic data
                err_ctx = y_hat[:, nin:ctx_len, :] - y_t[:, nin:, :]
                err_synt = y_sim - y_hat[:, ctx_len:, :]
                loss_ctx = torch.mean(err_ctx ** 2 / sigmay_train ** 2)
                loss_sim = torch.mean(err_synt[:,nin:,:] ** 2 / (sigmay_train ** 2 if regularization else sigmay_sim[:,nin:,:] ** 2))
                loss = (loss_ctx + gamma * loss_sim) / (1 + gamma )
                
               
                # Backpropagation and optimization
                loss.backward()
                # Reset gradients
                optimizer.step()



                with torch.no_grad():
                    
                    # Evaluate performance on training, validation, and test sets
                    # losses over the different number of the trajectories
                    test_loss = []
                    y_hat_t = []
                    train_loss = []
                    y_hat_v = []
                    val_loss = []
                    ts_loss = []
                    val_r2 = []
                    y_hat_test = []
                    test_loss = []
                    test_r2 = []
                    # for the different number of trajectories in the test
                    for ind, test_case in enumerate(test):  
                        n = test_case.state_initialization_window_length
                        # print(n)
                        y_model_s = apply_model(model, u_scaler, y_scaler, test_case.u, test_case.y[:n], train[ind].u)
                        y_t_model_s = y_model_s[:ctx_len,:]
                        y_v_model_s = y_model_s[ctx_len:ctx_len+val_len,:]
                        y_test_model_s = y_model_s[ctx_len+val_len:,:]
                        y_train = train[ind].y[:ctx_len,:]
                        y_val = train[ind].y[ctx_len:,:]
                        y_test = test_case.y
                        y_hat_t.append(y_t_model_s)
                        y_hat_v.append(y_v_model_s)
                        y_hat_test.append(y_test_model_s)
                        train_loss.append(RMSE(y_train, y_t_model_s)[0])
                        val_loss.append(RMSE(y_val, y_v_model_s)[0])
                        val_r2.append(R_squared(y_val, y_v_model_s)[0])
                        test_loss.append(RMSE(y_test[n:, :], y_test_model_s[n:, :])[0])
                        test_r2.append(R_squared(y_test[n:, :], y_test_model_s[n:, :])[0])
                    train_loss = np.mean(train_loss).tolist()
                    val_loss = np.mean(val_loss).tolist()
                    ts_loss = np.mean(test_loss).tolist()
                    val_r2 = np.mean(val_r2).tolist()


                    total_losses_row.append(loss.item())
                    total_v_losses_row.append(val_loss)
                    total_ts_losses_row.append(ts_loss)
                    # total_v_losses_2_row.append(val_loss[1]) ##for the validation loss for the second trajectory
                    total_test_losses_row.append(test_loss)
                    # total_test_losses_2_row.append(test_loss[1]) ##for the test loss for the second trajectory
                    total_loss_sim.append(loss_sim)
                    total_loss_ctx.append(loss_ctx)

                    # Update best validation loss
                    if val_loss < val_loss_best:
                    # idx_best = epoch  ## the epoch with smaller validation loss
                        train_loss_best = train_loss # as a best train loss, we take the one reached with early stopping
                        val_loss_best = val_loss
                        ts_loss_best = ts_loss
                        test_loss_best = test_loss
                        val_r2_best = val_r2
                        test_r2_best = test_r2

                    # Print progress
                    if epoch % print_frequency == 0:                        
                        # print(f'Epoch {epoch} | Train Loss {loss:.6f} | Loss_ctx {loss_ctx:.4f} |Loss_sim {loss_sim:.4f} | Validation Loss 1 {val_loss[0]:.4f}| Validation Loss 2 {val_loss[1]:.4f} | Test Loss 1 {test_loss[0]:.4f} |Test Loss 2 {test_loss[1]:.4f}')
                        print(f'Epoch {epoch} | Train Loss {loss:.6f} | Loss_ctx {loss_ctx:.4f} | Validation Loss {val_loss:.4f} | Test Loss 1 {test_loss[0]:.4f} |Test Loss 2 {test_loss[1]:.4f}')

        total_losses.append(total_losses_row)
        total_v_losses.append(total_v_losses_row)
        # total_v_losses_2.append(total_v_losses_2_row)
        total_test_losses.append(total_test_losses_row)
        total_ts_losses.append(total_ts_losses_row)
        # total_test_losses_2.append(total_test_losses_2_row)

        # Store best losses for each gamma
        train_loss_dict[gamma] = train_loss_best#loss.item()
        ts_loss_dict[gamma] = ts_loss_best
        val_loss_dict[gamma] = val_loss_best
        test_loss_dict[gamma] = test_loss_best
        val_r2_dict[gamma] = val_r2_best
        test_r2_dict[gamma] = test_r2_best
        print(test_loss_dict)


        print("\n")

        print("Train Losses:")
        for key, value in train_loss_dict.items():
            print(f"{key}: {value:.4f}")

        print("\n Validation Losses:")
        for key, value in val_loss_dict.items():
            print(f"{key}: {value:.4f}")

        print("\nTest Losses:")
        for key, value in test_loss_dict.items():
            if isinstance(value, list):
                formatted_values = ", ".join(f"{v:.4f}" for v in value)
                print(f"{key}: [{formatted_values}]")
    
        print("\n")
    
    # labels = [f'Gamma {i}' for i in gamma_list]
    # colors2 = plt.cm.viridis(np.linspace(0, 1, len(total_losses)))  # Use a color map for distinct colors
    # colors = ['blue', 'orange', 'green', 'red']
    
    # plt.gca().set_prop_cycle(color=colors2)
    # plt.figure()
    # plt.plot(np.array(total_losses).T)
    # plt.legend(labels)
    # plt.savefig(f'fig/train_loss_over_gamma_{lr}.png')
    # plt.figure()
    # plt.plot(np.array(total_v_losses).T)
    # plt.vlines(x=idx_best, ymin=min(map(min, total_v_losses)), ymax=max(map(max, total_v_losses)), colors=colors, linestyles='--', linewidth=2)
    # plt.legend(labels)
    # plt.savefig(f'fig/val_loss_over_gamma_{lr}.png')
    # plt.figure()
    # plt.plot(np.array(total_test_losses[0]).T)
    # plt.vlines(x=idx_best, ymin=min(map(min, total_test_losses[0])), ymax=max(map(max, total_test_losses[0])), colors=colors, linestyles='--', linewidth=2)
    # plt.legend(labels)
    # plt.savefig(f'fig/test_loss_1_over_gamma_{lr}.png')
    # plt.figure()
    # plt.plot(np.array(total_test_losses[1]).T)
    # plt.vlines(x=idx_best, ymin=min(map(min, total_test_losses[1])), ymax=max(map(max, total_test_losses[1])), colors=colors, linestyles='--', linewidth=2)
    # plt.legend(labels)
    # plt.savefig(f'fig/test_loss_2_over_gamma_{lr}.png')
    return train_loss_dict, val_loss_dict, test_loss_dict,ts_loss_dict, val_r2_dict, test_r2_dict




def apply_model(model,u_scaler,y_scaler, u_test, y_in, u_train = None):

    if config['simulate_train']:
        # Concatenate training and testing inputs for the model in case of training simulation
        u_conct = np.concatenate((u_scaler.transform(u_train), u_scaler.transform(u_test)))
        y_hat_conc = model(torch.tensor(u_conct).unsqueeze(0).float()) #.squeeze(dim = 0)

        # Append outputs after the training portion
        y_hat_norm = y_hat_conc[[0], :,:]

        # Convert normalized model outputs back to original scale for evaluation
            

    else:
        u_test = u_scaler.transform(u_test)
        y_hat_norm = model(torch.tensor(u_test).unsqueeze(0).float())
    

    y_hat = y_scaler.inverse_transform(y_hat_norm) 

    return y_hat.squeeze(0).detach().numpy()


def data_normalizer(train):
   
    train_norm = copy.deepcopy(train)

    u_scaler = Scaler()
    y_scaler = Scaler()

    # Concatenate all u and y data from the train set for fitting scalers

    conc_u_train = np.concatenate([t.u for t in train])
    conc_y_train = np.concatenate([t.y for t in train])
                                  
    u_scaler.fit(conc_u_train)
    y_scaler.fit(conc_y_train)

    # Transform each train data point using the fitted scalers
    for ind in range(len(train)):
        train_norm[ind].u = u_scaler.transform(train[ind].u)
        train_norm[ind].y  = y_scaler.transform(train[ind].y)
    

    return train_norm, u_scaler, y_scaler


if __name__ == '__main__':

    config = get_config()
    config['hidden_size'] = 20
    # Setting up basic parameters for data generation and training
    batch_size = 1
    nu, ny = 1, 1  # Dimensions of input and output
    nin = 20  # Number of initial steps to discard
    sigmay_train = 0.1  # Standard deviation of noise in training data
    n_MC = 50 # number of Monte Carlo runs

    # Define gamma values for regularization
    max_gamma = 5
    gamma_list = np.concatenate(([0], np.logspace(-3, np.log10(max_gamma), 7))).tolist()

    # Training hyperparameters
    lr = 1e-3
    max_epochs = 6001#config['max_epochs']
    print_frequency = 3000  # Frequency of printing training progress


    train_loss_list = []
    val_loss_list = []
    ts_loss_list = []
    test_loss_list=[] # at each monte carlo run, add performance score in lists
    val_r2_list = []
    test_r2_list=[] # at each monte carlo run, add performance score in lists

    train_loss_gamma0 = []
    val_loss_gamma0 = []
    ts_loss_gamma0 = []
    test_loss_gamma0 = []
    val_r2_gamma0 = []
    test_r2_gamma0 = []

    val_loss_best = []
    test_loss_best = []
    val_r2_best = []
    test_r2_best = []

    fig_path = Path("fig")
    fig_path.mkdir(exist_ok=True)

    # Compute settings and model configuration
    out_dir = "models"  # Output directory
    cuda_device = "cuda:0"
    no_cuda = False
    threads = 5
    torch.set_num_threads(threads)
    use_cuda = not no_cuda and torch.cuda.is_available()
    device_name = cuda_device if use_cuda else "cpu"
    device = torch.device(device_name)
    torch.set_float32_matmul_precision("high")

    # Loading pre-trained model
    out_dir = Path(out_dir)
    exp_data = torch.load(out_dir / "synthetic_CED_ckpt_430_fine-tuned_lower_noise.pt", map_location=device,
                          weights_only=False)
    cfg = exp_data["cfg"]

    # Handling potential missing attribute in configuration
    try:
        cfg.seed
    except AttributeError:
        cfg.seed = None


    i=0 #iteration number
    # I hope code with the right normalization
    for  np_seed, torch_seed in zip( np.arange(100,100+n_MC), np.arange(42,42+n_MC)):
        i+=1
        # print(i) ## for printing the iteration number of the MonteCarlo
        set_seed(np_seed,torch_seed)

        # Creation of the datasets
        train, test = config['command_load'](atleast_2d=True, always_return_tuples_of_datasets=True)
        

        # Initializing the model with loaded configuration
        model_args = exp_data["model_args"]
        conf = Config(**model_args)
        model = TSTransformer(conf).to(device)
        model.load_state_dict(exp_data["model"])
        

        # Train the model and record validation and test losses
        train_loss_dict, val_loss_dict, test_loss_dict,ts_loss_dict, val_r2_dict, test_r2_dict = train_dynoNet(test = test, train  = train,max_epochs=max_epochs,print_frequency=print_frequency, in_context_model=model,
                                                        sigmay_train=sigmay_train, gamma_list=gamma_list,
                                                        regularization=False)#,mean_std_y = mean_std_y)
        

        # # Find the gamma with the minimum value in val_loss_dict (excluding gamma=0)
        min_gamma = min((gamma for gamma in val_loss_dict if gamma != 0), key=val_loss_dict.get)

        train_loss_gamma0.append(train_loss_dict[0])
        ts_loss_gamma0.append(ts_loss_dict[0])
        val_loss_gamma0.append(val_loss_dict[0])
        test_loss_gamma0.append(test_loss_dict[0])
        val_r2_gamma0.append(val_r2_dict[0])
        test_r2_gamma0.append(test_r2_dict[0])

        val_loss_best.append(val_loss_dict[min_gamma])
        test_loss_best.append(test_loss_dict[min_gamma])
        val_r2_best.append(val_r2_dict[min_gamma])
        test_r2_best.append(test_r2_dict[min_gamma])

        train_loss_list.append(train_loss_dict)
        ts_loss_list.append(ts_loss_dict)
        val_loss_list.append(val_loss_dict)
        test_loss_list.append(test_loss_dict)
        val_r2_list.append(val_r2_dict)
        test_r2_list.append(test_r2_dict)

        # # Output the results of training
        # print('gamma vs validation loss')
        # print(val_loss_dict)
        # print('gamma vs test loss')
        # print(test_loss_dict)
        # print('gamma vs validation R2')
        # print(val_r2_dict)
        # print('gamma vs test R2')
        # print(test_r2_dict)
        # # plots


    # Extracting values for each gamma
    gamma_values = list(val_loss_list[0].keys())
    gamma_data = {gamma: [] for gamma in gamma_values}
    gamma_data_train = {gamma: [] for gamma in gamma_values}
    gamma_data_ts = {gamma: [] for gamma in gamma_values}
    gamma_data_test = {gamma: [] for gamma in gamma_values}


    for train_loss_dict in train_loss_list:
        for gamma, train_loss in train_loss_dict.items():
            gamma_data_train[gamma].append(train_loss)

    for val_loss_dict in val_loss_list:
        for gamma, val_loss in val_loss_dict.items():
            gamma_data[gamma].append(val_loss)

    for test_loss_dict in test_loss_list:
        for gamma, test_loss in test_loss_dict.items():
            gamma_data_test[gamma].append(test_loss)

    for ts_loss_dict in ts_loss_list:
        for gamma, ts_loss in ts_loss_dict.items():
            gamma_data_ts[gamma].append(ts_loss)

    # save data
    test_r2_gamma0 = np.nan_to_num(test_r2_gamma0, nan=0)
    data_to_save = {
        'train_loss_gamma0': train_loss_gamma0,
        'val_loss_gamma0': val_loss_gamma0,
        'test_loss_gamma0': test_loss_gamma0,
        'val_r2_gamma0': val_r2_gamma0,
        'test_r2_gamma0': test_r2_gamma0,
        'val_loss_best': val_loss_best,
        'test_loss_best': test_loss_best,
        'val_r2_best': val_r2_best,
        'test_r2_best': test_r2_best,
        'gamma_values' : gamma_values,
        'gamma_data': gamma_data,
        'gamma_data_train': gamma_data_train,
        'gamma_data_test':gamma_data_test
    }

    # Create subfolder
    folder_name = 'saved_data'
    os.makedirs(folder_name, exist_ok=True)

    # Generate filename with date and time
    current_time = datetime.datetime.now()
    filename = current_time.strftime("%Y-%m-%d_%H-%M-%S") + f"20orderlin_10innerchannel"
    file_path = os.path.join(folder_name, filename)

    # Save data using pickle
    with open(file_path, 'wb') as fp:
        pickle.dump(data_to_save, fp)



    ######################3
    # In running the script, you also specify the name of the pickle file where your data is saved



    ###########################3
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 6))
    # Boxplot for validation data

    axes[0].set_title('Training')
    for i, gamma in enumerate(gamma_values):
        if i == 0:
            axes[0].boxplot(gamma_data_train[gamma], positions=[i], widths=0.6, labels=["0"])
        elif i == len(gamma_values)-1:
            axes[0].boxplot(gamma_data_train[gamma], positions=[i], widths=0.6, labels=[f"{max_gamma}"])
        else:
            axes[0].boxplot(gamma_data_train[gamma], positions=[i], widths=0.6, labels=[f"{gamma:.3f}"])
    axes[0].set_xlabel('Gamma Values')
    axes[0].set_ylabel('Loss Values')
    axes[0].set_ylim(-0.1, 1.2)
    #plt.ylim(-0.1, 20)
    axes[0].grid(True)

    axes[1].set_title('Validation')
    for i, gamma in enumerate(gamma_values):
        if i==0:
            axes[1].boxplot(gamma_data[gamma], positions=[i], widths=0.6, labels=["0"])
        # elif i == len(gamma_values)-1:
        #     axes[1].boxplot(gamma_data[gamma], positions=[i], widths=0.6, labels=[f"{max_gamma}"])
        else:
            axes[1].boxplot(gamma_data[gamma], positions=[i], widths=0.6, labels=[f"{gamma:.3f}"])
    axes[1].set_xlabel('Gamma Values')
    axes[1].set_ylabel('Loss Values')
    axes[1].set_ylim(-0.1, 1.2)
    #plt.ylim(-0.1, 20)
    axes[1].grid(True)

    axes[2].set_title('TS')
    print(gamma_data_ts)
    print(gamma_data)
    for i, gamma in enumerate(gamma_data_ts):
        if i==0:
            axes[2].boxplot(gamma_data_ts[gamma], positions=[i], widths=0.6, labels=["0"])
        elif i == len(gamma_values)-1:
            axes[2].boxplot(gamma_data_ts[gamma], positions=[i], widths=0.6, labels=[f"{max_gamma}"])
        else:
            axes[2].boxplot(gamma_data_ts[gamma], positions=[i], widths=0.6, labels=[f"{gamma:.3f}"])
    axes[2].set_xlabel('Gamma Values')
    axes[2].set_ylabel('Loss Values')
    axes[2].set_ylim(-0.1, 1.2)
    #plt.ylim(-0.1, 20)
    axes[2].grid(True)

    

    # Boxplot for test data
    axes[3].set_title('Test 1')
    axes[4].set_title('Test 2')
    # print(gamma_data_test)
    for i, gamma in enumerate(gamma_values):
        array = np.array(gamma_data_test[gamma])
        transposed = array.T  # or np.transpose(array)
        transposed.tolist()
        print(transposed)
        if i==0:
            axes[3].boxplot(transposed[0], positions=[i], widths=0.6, labels=["0"])
            axes[4].boxplot(transposed[1], positions=[i], widths=0.6, labels=["0"])
        elif i == len(gamma_values)-1:
            axes[3].boxplot(transposed[0], positions=[i], widths=0.6, labels=[f"{max_gamma}"])
            axes[4].boxplot(transposed[1], positions=[i], widths=0.6, labels=[f"{max_gamma}"])
        else:
            axes[3].boxplot(transposed[0], positions=[i], widths=0.6, labels=[f"{gamma:.3f}"])
            axes[4].boxplot(transposed[1], positions=[i], widths=0.6, labels=[f"{gamma:.3f}"])
    axes[3].set_xlabel('Gamma Values')
    axes[3].set_ylabel('Loss Values')
    axes[3].set_ylim(-0.1, 1.2)
    axes[3].grid(True)
    axes[4].set_xlabel('Gamma Values')
    axes[4].set_ylabel('Loss Values')
    axes[4].set_ylim(-0.1, 1.2)
    axes[4].grid(True)

    #plt.tight_layout()
    # plt.ylim(-0.1, 20)
    plt.savefig('fig/boxplot_losses_CED_prob.png')
    plt.show()


    # # # Creating box plots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))

    # Boxplot for validation data
    axes[0].boxplot(gamma_data[0], positions=[0], widths=0.6, labels=['gamma validation= 0'])
    axes[0].boxplot(val_loss_best, positions=[1], widths=0.6, labels=['best validation gamma ≠ 0'])
    axes[0].set_title('Validation Losses')
    axes[0].set_ylim(-0.1, 1)
    axes[0].grid(True)

    # Boxplot for test data
    array1 = np.array(gamma_data_test[0])
    transposed1 = array1.T  # or np.transpose(array)
    transposed1.tolist()
    array = np.array(test_loss_best)
    transposed = array.T  # or np.transpose(array)
    transposed.tolist()

    axes[1].boxplot(transposed1[0], positions=[0], widths=0.6, labels=['gamma test= 0'])
    axes[1].boxplot(transposed[0], positions=[1], widths=0.6, labels=['best test gamma ≠ 0'])
    axes[1].set_title('Test Losses')
    axes[1].set_ylim(-0.1, 1)
    axes[1].grid(True)

    axes[2].boxplot(transposed1[1], positions=[0], widths=0.6, labels=['gamma test= 0'])
    axes[2].boxplot(transposed[1], positions=[1], widths=0.6, labels=['best test gamma ≠ 0'])
    axes[2].set_title('Test Losses')
    axes[2].grid(True)

    #plt.tight_layout()
    plt.ylim(-0.1, 1)
    plt.savefig('fig/boxplot_val_test_CED_prob.png')
    plt.show()


    # R2

    # Creating box plots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))

    # Boxplot for validation data
    axes[0].boxplot(val_r2_gamma0, positions=[0], widths=0.6, labels=['gamma validation= 0'])
    axes[0].boxplot(val_r2_best, positions=[1], widths=0.6, labels=['best validation gamma ≠ 0'])
    axes[0].set_title('R2 val')
    axes[0].grid(True)

    # Boxplot for test data
    array1 = np.array(test_r2_gamma0)
    transposed1 = array1.T  # or np.transpose(array)
    transposed1.tolist()
    array = np.array(test_r2_best)
    transposed = array.T  # or np.transpose(array)
    transposed.tolist()
    axes[1].boxplot(transposed1[0], positions=[0], widths=0.6, labels=['gamma test= 0'])
    axes[1].boxplot(transposed[0], positions=[1], widths=0.6, labels=['best test gamma ≠ 0'])
    axes[1].set_title('R2 test')
    axes[1].grid(True)

    axes[2].boxplot(transposed1[1], positions=[0], widths=0.6, labels=['gamma test= 0'])
    axes[2].boxplot(transposed[1], positions=[1], widths=0.6, labels=['best test gamma ≠ 0'])
    axes[2].set_title('R2 test')
    axes[0].set_ylim(-0.1, 1)
    axes[1].set_ylim(-0.1, 1)
    axes[2].set_ylim(-0.1, 1)
    
    axes[2].grid(True)

    # plt.tight_layout()
    plt.ylim(-0.1, 1)
    plt.savefig('fig/boxplot_r2_val_test_prob.png')
    plt.show()