import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from pathlib import Path

from data_prep import get_loaders
from model_Unet import UNet


# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
CHECKPOINT_DIR = "checkpoints"
LOG_FILE = "training_log.txt"

# Create directory for saving weights
Path(CHECKPOINT_DIR).mkdir(exist_ok=True)


def log_message(message, log_file=LOG_FILE):
    """Prints message to console and logs to file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(full_message + "\n")


def train_epoch(model, train_loader, optimizer, loss_fn, device):
    """Single training epoch"""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, masks.unsqueeze(1))  # add channel dimension for mask
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Log batch progress
        if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
            avg_loss = total_loss / (batch_idx + 1)
            log_message(
                f"  Batch [{batch_idx + 1}/{len(train_loader)}] - Loss: {loss.item():.6f} | "
                f"Avg Loss: {avg_loss:.6f}"
            )
    
    return total_loss / len(train_loader)


def validate(model, val_loader, loss_fn, device):
    """Single validation epoch"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, masks.unsqueeze(1))
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir=CHECKPOINT_DIR):
    """Save model weights"""
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch:03d}_loss_{loss:.6f}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    log_message(f"✓ Model saved: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load model weights"""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    
    log_message(f"✓ Model loaded: {checkpoint_path} (epoch {epoch})")
    return epoch, loss


def train_model(num_epochs=EPOCHS, resume_from=None):
    """Main training loop"""
    
    log_message("=" * 80)
    log_message("U-Net TRAINING STARTED for oil spill segmentation")
    log_message("=" * 80)
    log_message(f"Device: {DEVICE}")
    log_message(f"Total epochs: {num_epochs}")
    log_message(f"Batch size: {BATCH_SIZE}")
    log_message(f"Learning rate: {LEARNING_RATE}")
    log_message("")
    
    # Load data
    log_message("Loading data...")
    train_loader, val_loader, test_loader = get_loaders(batch_size=BATCH_SIZE)
    log_message(f"✓ Data loaded")
    log_message("")
    
    # Initialize model
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    log_message(f"U-Net model initialized")
    log_message(f"Total number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    log_message("")
    
    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()  # for binary segmentation
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5
    )
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume_from and os.path.exists(resume_from):
        start_epoch, best_val_loss = load_checkpoint(resume_from, model, optimizer)
        log_message("")
    
    # Main training loop
    log_message("TRAINING STARTED")
    log_message("-" * 80)
    
    for epoch in range(start_epoch, num_epochs):
        epoch_num = epoch + 1
        log_message(f"\n[Epoch {epoch_num}/{num_epochs}]")
        
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        log_message(f"  Training - Average loss: {train_loss:.6f}")
        
        # Validation
        val_loss = validate(model, val_loader, loss_fn, DEVICE)
        log_message(f"  Validation - Average loss: {val_loss:.6f}")
        
        # LR Scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        log_message(f"  Learning Rate: {current_lr:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            log_message(f"  🎯 IMPROVEMENT! (previous: {best_val_loss:.6f})")
            save_checkpoint(model, optimizer, epoch_num, val_loss)
        else:
            log_message(f"  ⚠ No improvement (best: {best_val_loss:.6f})")
    
    log_message("\n" + "=" * 80)
    log_message("TRAINING COMPLETED")
    log_message(f"Best validation loss: {best_val_loss:.6f}")
    log_message("=" * 80)


if __name__ == "__main__":
    # Clear previous log (optional)
    # if os.path.exists(LOG_FILE):
    #     os.remove(LOG_FILE)
    
    # Run training
    train_model(num_epochs=EPOCHS)
    
    # To resume training from checkpoint:
    # train_model(num_epochs=EPOCHS, resume_from="checkpoints/model_epoch_010_loss_0.123456.pth")
