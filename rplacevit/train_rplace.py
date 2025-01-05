import argparse
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import AdamW
from torch.nn.parallel import DataParallel
from model import RPlaceTimeTransformer, RPlaceColorTransformerV2
from dataset import RPlaceColorDataset, RPlaceTimeDataset
from train import train_model
from utils import trainable_parameters

def configure_arg_parser():
    """
    Configure the argument parser for training R/Place models.
    
    Returns:
        parser: The configured argparse parser.
    """
    parser = argparse.ArgumentParser(description="Train R/Place models")
    parser.add_argument("--model_type", type=str, choices=["color", "time"], required=True, help="Type of model to train")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of epochs to train")
    parser.add_argument("--model_dim", type=int, default=512, help="Dimension of the model")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_blocks", type=int, default=12, help="Number of transformer blocks")
    parser.add_argument("--ff_dim", type=int, default=2048, help="Dimension of the feed-forward network")
    parser.add_argument("--dropout_rate", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--center_size", type=int, default=16, help="Width of the center view")
    parser.add_argument("--peripheral_size", type=int, default=64, help="Width of the peripheral view")
    parser.add_argument("--use_peripheral", action="store_true", help="Use peripheral view")
    parser.add_argument("--output_strategy", type=str, default="cls_token", choices=["cls_token", "avg", "center"], help="Output strategy for color model")
    parser.add_argument("--single_partition", type=int, default=None, help="Single partition to use")
    parser.add_argument("--use_user_features", action="store_true", help="Use user features")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--min_x", type=int, default=0, help="Minimum x-coordinate of the region of interest")
    parser.add_argument("--min_y", type=int, default=0, help="Minimum y-coordinate of the region of interest")
    parser.add_argument("--max_x", type=int, default=2048, help="Maximum x-coordinate of the region of interest")
    parser.add_argument("--max_y", type=int, default=2048, help="Maximum y-coordinate of the region of interest")
    
    return parser

def initialize_training_environment(args):
    """
    Initialize the training environment for the model.
    
    Args:
        args: Parsed arguments from argparse
        
    Returns:
        model: The model to train
        dataloader: DataLoader for the training dataset
        loss_fn: Loss function
        optimizer: Optimizer for training
        device: Device to use for training
        checkpoint_dir: Directory to save model checkpoints
        save_interval: Number of steps to save a checkpoint
        log_interval: Number of steps to log training information
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPUs")
    else:
        device = torch.device("cpu")
        num_gpus = 1
        print("CUDA is not available. Using CPU")

    if args.model_type == "color":
        dataset = RPlaceColorDataset(
            single_partition=args.single_partition,
            use_user_features=args.use_user_features,
            view_width=args.center_size,
            min_x=args.min_x,
            min_y=args.min_y,
            max_x=args.max_x,
            max_y=args.max_y
        )
        loss_fn = CrossEntropyLoss()
    else:
        dataset = RPlaceTimeDataset(
            single_partition=args.single_partition,
            use_user_features=args.use_user_features,
            view_width=args.center_size,
            min_x=args.min_x,
            min_y=args.min_y,
            max_x=args.max_x,
            max_y=args.max_y
        )
        loss_fn = MSELoss()

    dataset.load_end_states()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model_config = {
        "in_channels": 32,
        "d_model": args.model_dim,
        "num_heads": args.num_heads,
        "num_blocks": args.num_blocks,
        "d_ff": args.ff_dim,
        "dropout": args.dropout_rate,
        "center_width": args.center_size,
        "peripheral_width": args.peripheral_size,
        "use_peripheral": args.use_peripheral,
    }

    if args.model_type == "color":
        model = RPlaceColorTransformerV2(**model_config, output_strategy=args.output_strategy)
    else:
        model = RPlaceTimeTransformer(**model_config)

    model.to(device)
    model.init_weights()

    if num_gpus > 1:
        model = DataParallel(model)

    print(f"Trainable parameters: {trainable_parameters(model)}")

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    checkpoint_dir = f"./{args.model_type}_model_checkpoints"
    save_interval = 1000
    log_interval = 100

    return model, dataloader, loss_fn, optimizer, device, checkpoint_dir, save_interval, log_interval

def execute_training():
    """
    Execute the training process for the R/Place model.
    
    Returns:
        trained_model: The trained model
        training_losses: List of training losses
    """
    parser = configure_arg_parser()
    args = parser.parse_args()

    model, dataloader, loss_fn, optimizer, device, checkpoint_dir, save_interval, log_interval = initialize_training_environment(args)

    training_losses, total_time = train_model(
        model=model,
        train_loader=dataloader,
        criterion=loss_fn,
        optimizer=optimizer,
        device=device,
        model_path=checkpoint_dir,
        num_epochs=args.num_epochs,
        save_every=save_interval,
        log_every=log_interval,
        checkpoint_path=args.checkpoint_path,
    )

    print(f"Training completed in {total_time:.2f} seconds")
    return model, training_losses

if __name__ == "__main__":
    trained_model, losses = execute_training()