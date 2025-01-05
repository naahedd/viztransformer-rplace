import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_training_loss(checkpoint_file, smoothing_window=100):
    """
    Visualize the training losses from a checkpoint file by plotting them.
    
    Args:
        checkpoint_file (str): Path to the checkpoint file containing loss data.
        smoothing_window (int): Number of steps to average over for smoothing the loss curve (default: 100).
    """
    # Load the checkpoint file
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    
    # Retrieve the loss data from the checkpoint
    loss_data = checkpoint.get('losses', [])
    
    # Check if loss data is available
    if not loss_data:
        print("No loss data found in the checkpoint.")
        return
    
    # Smooth the loss data by averaging over the specified window
    smoothed_losses = []
    for i in range(0, len(loss_data), smoothing_window):
        smoothed_losses.append(np.mean(loss_data[i:i+smoothing_window]))
    
    # Generate x-axis steps for plotting
    steps = list(range(0, len(loss_data), smoothing_window))
    
    # Plot the smoothed loss curve
    plt.figure(figsize=(12, 6))
    plt.plot(steps, smoothed_losses, label='Smoothed Loss')
    plt.title(f'Training Loss Over Time (Smoothed every {smoothing_window} steps)')
    plt.xlabel('Training Step')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.legend()
    
    # Save the plot as an image file
    output_image = 'training_loss_plot.png'
    plt.savefig(output_image)
    print(f"Training loss plot saved as {output_image}")

if __name__ == "__main__":
    # Set up argument parser for command-line interface
    parser = argparse.ArgumentParser(description="Visualize training losses from a checkpoint file.")
    parser.add_argument("checkpoint_file", type=str, help="Path to the checkpoint file containing loss data.")
    parser.add_argument("--smoothing_window", type=int, default=100, help="Number of steps to average over for smoothing (default: 100).")
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Call the function to visualize training losses
    visualize_training_loss(args.checkpoint_file, args.smoothing_window)