import os
import time
import torch
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from collections import deque

def train_single_epoch(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    gradient_scaler: GradScaler,
    current_epoch: int,
    total_epochs: int,
    global_step_counter: int,
    recent_loss_history: deque,
    checkpoint_interval: int,
    logging_interval: int,
    checkpoint_directory: str,
    loss_tracker: list
) -> tuple:
    """
    Train the model for a single epoch.
    
    Args:
        model (torch.nn.Module): The model to train.
        data_loader (torch.utils.data.DataLoader): The training data loader.
        loss_function (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (str): The device to train on.
        gradient_scaler (GradScaler): The gradient scaler for mixed precision training.
        current_epoch (int): The current epoch number.
        total_epochs (int): The total number of epochs.
        global_step_counter (int): The current global step counter.
        recent_loss_history (deque): A deque to store recent losses for averaging.
        checkpoint_interval (int): Save a checkpoint every n steps.
        logging_interval (int): Log the loss every n steps.
        checkpoint_directory (str): The directory to save model checkpoints.
        loss_tracker (list): A list to store all losses.
    """
    model.train()
    progress_bar = tqdm(data_loader, desc=f"Epoch {current_epoch + 1}/{total_epochs}")
    
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        with autocast('cuda'):
            predictions = model(inputs)
            loss = loss_function(predictions, targets)

        gradient_scaler.scale(loss).backward()
        clip_grad_norm_(model.parameters(), 1.0)
        gradient_scaler.step(optimizer)
        gradient_scaler.update()

        global_step_counter += 1
        recent_loss_history.append(loss.item())
        loss_tracker.append(loss.item())
        average_loss = sum(recent_loss_history) / len(recent_loss_history)

        progress_bar.set_postfix_str(f"Loss: {loss.item():.4f}, Avg Loss: {average_loss:.4f}")

        if global_step_counter % logging_interval == 0:
            print(f"Epoch {current_epoch + 1}, Step {global_step_counter}: Loss: {loss.item():.4f}, Avg Loss: {average_loss:.4f}")

        if checkpoint_interval != 0 and global_step_counter % checkpoint_interval == 0:
            save_model_checkpoint(model, optimizer, gradient_scaler, current_epoch, global_step_counter, recent_loss_history, checkpoint_directory, loss_tracker)

    return global_step_counter, recent_loss_history, loss_tracker

def save_model_checkpoint(model, optimizer, gradient_scaler, current_epoch, global_step_counter, recent_loss_history, checkpoint_directory, loss_tracker):
    """
    Save a checkpoint of the model.
    
    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        gradient_scaler (GradScaler): The gradient scaler to save.
        current_epoch (int): The current epoch number.
        global_step_counter (int): The current global step counter.
        recent_loss_history (deque): A deque containing recent losses.
        checkpoint_directory (str): The directory to save the checkpoint.
        loss_tracker (list): A list containing all losses.
    """
    if not os.path.exists(checkpoint_directory):
        os.makedirs(checkpoint_directory)
    checkpoint = {
        'epoch': current_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'gradient_scaler': gradient_scaler.state_dict(),
        'global_step': global_step_counter,
        'recent_losses': list(recent_loss_history),
        'losses': loss_tracker
    }
    checkpoint_path = os.path.join(checkpoint_directory, f"checkpoint_epoch_{current_epoch + 1}_step_{global_step_counter}.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {current_epoch + 1}, step {global_step_counter}! Avg loss over last {len(recent_loss_history)} steps: {sum(recent_loss_history)/len(recent_loss_history):.4f}")

def execute_training(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        loss_function: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        checkpoint_directory: str,
        total_epochs: int,
        checkpoint_interval: int = 0,
        logging_interval: int = 100,
        resume_checkpoint_path: str | None = None,
) -> tuple:
    """
    Execute the training process for the model.
    
    Args:
        model (torch.nn.Module): The model to train.
        data_loader (torch.utils.data.DataLoader): The training data loader.
        loss_function (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (str): The device to train on.
        checkpoint_directory (str): The directory to save checkpoints.
        total_epochs (int): The total number of epochs to train.
        checkpoint_interval (int): Save a checkpoint every n steps.
        logging_interval (int): Log the loss every n steps.
        resume_checkpoint_path (str): Path to a checkpoint to resume training from.
    """
    training_losses = []
    start_time = time.time()
    global_step_counter = 0
    start_epoch = 0
    gradient_scaler = GradScaler()
    recent_loss_history = deque(maxlen=1000)
    loss_tracker = []

    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        gradient_scaler.load_state_dict(checkpoint['gradient_scaler'])
        global_step_counter = checkpoint['global_step']
        recent_loss_history = deque(checkpoint['recent_losses'], maxlen=1000)
        start_epoch = checkpoint['epoch'] + 1
        loss_tracker = checkpoint.get('losses', [])
        
        print(f"Resuming training from epoch {start_epoch}, step {global_step_counter}")
    elif resume_checkpoint_path:
        print(f"Checkpoint not found at {resume_checkpoint_path}, starting from scratch")

    for epoch in range(start_epoch, total_epochs):
        global_step_counter, recent_loss_history, loss_tracker = train_single_epoch(
            model, data_loader, loss_function, optimizer, device, gradient_scaler,
            epoch, total_epochs, global_step_counter, recent_loss_history, checkpoint_interval, logging_interval, checkpoint_directory, loss_tracker
        )
        epoch_loss = sum(recent_loss_history) / len(recent_loss_history)
        training_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{total_epochs} completed. Average loss: {epoch_loss:.4f}")

        if checkpoint_interval != 0:
            save_model_checkpoint(model, optimizer, gradient_scaler, epoch, global_step_counter, recent_loss_history, checkpoint_directory, loss_tracker)

    total_time = time.time() - start_time
    return training_losses, total_time, loss_tracker