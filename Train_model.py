"""
Authors: Antonio Rangel & Juan Terven
Date: 2023

Description:
This script trains a semantic segmentation model using either LRASPP or FCN architectures.
It supports HDF5 datasets and integrates with Weights & Biases (WandB) for experiment tracking.
The training pipeline includes performance monitoring, early stopping, and automatic model checkpointing.

Main Features:
- Initializes and trains a semantic segmentation model using LRASPP or FCN.
- Loads training and validation data from an HDF5 dataset.
- Supports logging to Weights & Biases (WandB) for experiment tracking.
- Implements batch-wise training with adaptive learning.
- Computes multiple evaluation metrics including Overall Accuracy (OA) and IoU.
- Saves model checkpoints based on validation performance.
- Uses early stopping to prevent unnecessary computations.
- Provides GPU selection and monitoring using `nvidia-smi`.
- Interactive CLI for model selection, hyperparameter configuration, and dataset specification.
"""

import os
from timeit import default_timer as timer
from tqdm import tqdm
from torchvision.models.segmentation import lraspp_mobilenet_v3_large,LRASPP_MobileNet_V3_Large_Weights,fcn_resnet50,FCN_ResNet50_Weights
from sklearn.metrics import  jaccard_score
from pathlib import Path
from datetime import datetime
import wandb
import subprocess
import torch
import h5py
from typing import List, Optional,Callable


def run_nvidia_smi():
    """
    Executes the `nvidia-smi` command to check GPU status.
    """
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("nvidia-smi Output:")
        print(result.stdout)
    except FileNotFoundError:
        print("The `nvidia-smi` command was not found. Ensure NVIDIA drivers are installed.")


def prompt_user(message="Enter a number: "):
    """
    Prompts the user for input.

    Args:
        message (str): Message to display to the user.

    Returns:
        str: User input.
    """
    return input(message)


def initialize_wandb(run_name):
    """
    Initializes a Weights & Biases (wandb) session.

    Args:
        run_name (str): Name for the wandb run.
    """
    try:
        entity = prompt_user("Enter the WandB entity: ")
        project = prompt_user("Enter the WandB project name: ")

        with open("keys.txt", 'r') as file:
            content = file.readlines()
            _, _, wandb_key = content[0].split(",")

        wandb.login(key=wandb_key.strip())
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        wandb.init(
            entity=entity,
            project=project,
            name=f"{current_date}_{run_name}"
        )
    except Exception as e:
        print(f"Error initializing wandb: {e}")


def initialize_model(model_type, device, num_classes=9, seed=42):
    """
    Initializes and configures a segmentation model.

    Args:
        model_type (str): Type of model to initialize ('LRASPP' or 'FCN').
        device (torch.device): Device to load the model on (CPU or GPU).
        num_classes (int): Number of classes for the model (default: 9).

    Returns:
        torch.nn.Module: Configured model.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if model_type == "LRASPP":
        model = lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT, num_classes=21)
        model.classifier.low_classifier = torch.nn.Conv2d(40, num_classes, kernel_size=1, stride=1)
        model.classifier.high_classifier = torch.nn.Conv2d(128, num_classes, kernel_size=1, stride=1)
        print("Loaded LRASPP model.")
    elif model_type == "FCN":
        model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT, num_classes=21)
        model.classifier[4] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        print("Loaded FCN model.")
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'LRASPP' or 'FCN'.")

    model.to(device)
    return model


def setup_environment_and_model():
    """
    Configures the environment and initializes the model.

    Returns:
        Tuple containing:
        - model (torch.nn.Module): Initialized model.
        - device (torch.device): Computation device.
        - num_epochs (int): Number of training epochs.
        - use_wandb (bool): Whether to log training to wandb.
        - train_data_path (str): Path to training data.
        - val_data_path (str): Path to validation data.
    """
    run_nvidia_smi()
    selected_gpu = prompt_user("Enter the GPU number to use: ")
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{selected_gpu}"
    device = torch.device("cuda:0")
    print(f"Using GPU {selected_gpu}...")

    wandb_response = prompt_user("Would you like to log progress to Wandb? (y/n): ").strip().lower()
    use_wandb = wandb_response in ['y', 'yes']

    print("Select the model to train:")
    print("0: LRASPP")
    print("1: FCN")
    model_option = prompt_user("Enter 0 or 1 for the desired model: ")
    model_type = 'LRASPP' if model_option == '0' else 'FCN' if model_option == '1' else 'LRASPP'

    while True:
        try:
            num_epochs = int(prompt_user("Enter the number of epochs for training: "))
            if num_epochs > 0:
                break
            else:
                print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    train_data_path = prompt_user("Enter the path to your training data: ")
    val_data_path = prompt_user("Enter the path to your validation data: ")

    if use_wandb:
        initialize_wandb(f"{model_type}_SIZE_256_BATCH_32_MOSAIC")

    model = initialize_model(model_type, device)

    return model, device, num_epochs, use_wandb, train_data_path, val_data_path


def overall_accuracy(pred, target):
    """
    Computes overall accuracy.
    """
    return torch.count_nonzero(target.squeeze() == pred.argmax(dim=1)) / torch.numel(target.squeeze())


def mean_iou(pred, target):
    """
    Computes mean Intersection over Union (IoU).
    """
    target_np = target.cpu().numpy().squeeze()
    pred_np = pred.argmax(dim=1).detach().cpu().numpy()
    return jaccard_score(target_np.reshape(-1), pred_np.reshape(-1), zero_division=1., average='macro')


def cross_entropy_loss(predictions, targets):
    """
    Computes cross-entropy loss.
    """
    return torch.nn.functional.cross_entropy(predictions, targets.squeeze())


def train_step_h5(
        hdf5_file,
        model: torch.nn.Module,
        loss_function: Callable,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        accuracy_metrics: Optional[List] = None
):
    """
    Performs a single training step using data from an HDF5 dataset.

    Args:
        hdf5_file: Open HDF5 file containing the dataset.
        model (torch.nn.Module): PyTorch model to be trained.
        loss_function (Callable): Loss function for optimization.
        optimizer (torch.optim.Optimizer): Optimizer to update the model's parameters.
        batch_size (int): Number of samples per batch.
        accuracy_metrics (Optional[List]): List of accuracy metric functions.

    Returns:
        float: Average training loss.
    """
    accumulated_loss = 0

    if accuracy_metrics is not None:
        accuracy_values = [0] * len(accuracy_metrics)

    model = model.cuda()
    total_samples = hdf5_file['masks'].shape[0]

    for start in tqdm(range(0, total_samples, batch_size), desc="Training..."):
        end = min(start + batch_size, total_samples)

        predictors_batch = hdf5_file['predictors'][start:end]
        masks_batch = hdf5_file['masks'][start:end]

        predictors = torch.tensor(predictors_batch, dtype=torch.float32).cuda(non_blocking=True)
        masks = torch.tensor(masks_batch, dtype=torch.long).cuda(non_blocking=True)

        predictions = model(predictors)['out']
        loss = loss_function(predictions, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accumulated_loss += float(loss) / (total_samples // batch_size)

        if accuracy_metrics is not None:
            for i, accuracy_function in enumerate(accuracy_metrics):
                accuracy_values[i] += float(accuracy_function(predictions, masks)) / (total_samples // batch_size)

    if accuracy_metrics is not None:
        return accumulated_loss, accuracy_values
    return accumulated_loss


def validation_step_h5(
        hdf5_file,
        model: torch.nn.Module,
        batch_size: int,
        accuracy_metrics: Optional[List] = None
):
    """
    Performs a single validation step using data from an HDF5 dataset.

    Args:
        hdf5_file: Open HDF5 file containing the dataset.
        model (torch.nn.Module): PyTorch model to be evaluated.
        batch_size (int): Number of samples per batch.
        accuracy_metrics (Optional[List]): List of accuracy metric functions.

    Returns:
        List[float]: List of computed accuracy metrics.
    """
    model = model.cuda()
    accuracy_values = [0.] * len(accuracy_metrics) if accuracy_metrics is not None else []

    with torch.no_grad():
        total_samples = hdf5_file['masks'].shape[0]

        for start in tqdm(range(0, total_samples, batch_size), desc="Validating..."):
            end = min(start + batch_size, total_samples)

            predictors_batch = hdf5_file['predictors'][start:end]
            masks_batch = hdf5_file['masks'][start:end]

            predictors = torch.tensor(predictors_batch, dtype=torch.float32).cuda(non_blocking=True)
            masks = torch.tensor(masks_batch, dtype=torch.long).cuda(non_blocking=True)

            predictions = model(predictors)['out']

            if accuracy_metrics is not None:
                for i, accuracy_function in enumerate(accuracy_metrics):
                    accuracy_values[i] += float(accuracy_function(predictions, masks)) / (total_samples // batch_size)

    return accuracy_values


def train(
        model: torch.nn.Module,
        training_data_path: str,
        batch_size: int,
        validation_data_path: str,
        optimizer: torch.optim.Optimizer,
        loss_function: Callable,
        accuracy_functions: Optional[List] = None,
        num_epochs: int = 5,
        enable_wandb: bool = False
):
    """
    Trains a segmentation model using HDF5 datasets.

    Args:
        model (torch.nn.Module): The model to be trained.
        training_data_path (str): Path to the training dataset in HDF5 format.
        batch_size (int): Batch size for training.
        validation_data_path (str): Path to the validation dataset in HDF5 format.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        loss_function (Callable): Loss function for optimization.
        accuracy_functions (Optional[List]): List of accuracy metrics.
        num_epochs (int): Number of training epochs.
        enable_wandb (bool): Whether to log the training process to Weights & Biases.

    Returns:
        dict: Dictionary containing training loss and validation metrics.
    """
    if enable_wandb:
        wandb.watch(model)

    training_results = {"train_loss": [], "val_OA": [], "val_IOU": []}
    best_validation_accuracy = 0.0
    patience_limit = 20
    patience_counter = 0

    training_file = h5py.File(training_data_path, 'r', libver='latest', swmr=True)
    validation_file = h5py.File(validation_data_path, 'r', libver='latest', swmr=True)

    for epoch in tqdm(range(num_epochs), desc="Training"):
        train_loss = train_step_h5(
            f=training_file,
            model=model,
            loss_fn=loss_function,
            optimizer=optimizer,
            batch_size=batch_size
        )

        validation_accuracy = validation_step_h5(
            f=validation_file,
            model=model,
            batch_size=batch_size,
            acc_fns=accuracy_functions
        )

        if enable_wandb:
            wandb.log({
                "Epoch": epoch + 1,
                "train_loss": train_loss,
                "val_OA": validation_accuracy[0],
                "val_IOU": validation_accuracy[1]
            })

        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch + 1} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Validation OA: {validation_accuracy[0]:.4f} | "
                f"Validation IOU: {validation_accuracy[1]:.4f}"
            )

        if validation_accuracy[0] > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy[0]
            patience_counter = 0
            model_save_path = Path("Checkpoints")
            model_save_path.mkdir(parents=True, exist_ok=True)
            checkpoint_name = f"Segmentation_Model_Epoch_{epoch}.pth"
            torch.save(model.state_dict(), model_save_path / checkpoint_name)
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            print(f"Stopping training due to no improvement in {patience_limit} epochs.")
            break

        training_results["train_loss"].append(train_loss)
        training_results["val_OA"].append(validation_accuracy[0])
        training_results["val_IOU"].append(validation_accuracy[1])

    final_checkpoint_name = "Segmentation_Model_Final.pth"
    torch.save(model.state_dict(), model_save_path / final_checkpoint_name)

    training_file.close()
    validation_file.close()

    return training_results


def main():
    """
    Main function to initialize the environment, configure the model, and train it.
    """
    # Initialize model, device, number of epochs, and dataset paths
    model, device, num_training_epochs, enable_wandb, training_dataset_path, validation_dataset_path = setup_environment_and_model()

    # Define the loss function and optimizer
    loss_function = cross_entropy_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Start the timer to measure training duration
    start_time = timer()

    # Train the model
    training_results = train(
        model=model,
        train_file=training_dataset_path,
        batch_size=64,
        val_file=validation_dataset_path,
        optimizer=optimizer,
        loss_fn=loss_function,
        epochs=num_training_epochs,
        acc_fns=[overall_accuracy, mean_iou],
    )

    # End the timer and display the total training duration
    end_time = timer()
    print(f"Total training time: {end_time - start_time:.3f} seconds")


if __name__ == "__main__":
    main()