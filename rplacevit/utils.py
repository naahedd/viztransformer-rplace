# import torch
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# def visualize_loss_curves(train_loss_history: list, val_loss_history: list, logging_interval: int):
#     """
#     Plot the training and validation loss curves.
    
#     Args:
#         train_loss_history (list): List of training losses.
#         val_loss_history (list): List of validation losses.
#         logging_interval (int): Interval at which losses were logged.
#     """
#     plt.figure(figsize=(12, 6))
#     steps = range(logging_interval, len(train_loss_history) * logging_interval + 1, logging_interval)
#     plt.plot(steps, train_loss_history, label='Training Loss')
#     plt.plot(steps, val_loss_history, label='Validation Loss')
#     plt.legend()
#     plt.xlabel('Steps')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Loss Over Steps')
#     plt.show()
