import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MNIST_DNN
import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import glob

def plot_metrics(train_losses, train_accs, val_losses, val_accs, save_path='training_metrics.png'):
    """Plot training and validation metrics."""
    plt.figure(figsize=(15, 6))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy', color='blue', linewidth=2)
    plt.plot(val_accs, label='Validation Accuracy', color='red', linewidth=2)
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(true_labels, predictions, save_path='confusion_matrix.png'):
    """Plot confusion matrix."""
    conf_matrix = torch.zeros(10, 10)
    for t, p in zip(true_labels, predictions):
        conf_matrix[t, p] += 1
        
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, cmap='Blues')
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Add numbers to the plot
    for i in range(10):
        for j in range(10):
            plt.text(j, i, int(conf_matrix[i, j]),
                    ha="center", va="center")
    
    plt.savefig(save_path)
    plt.close()

def get_train_transforms():
    """Get training data transformations with augmentation."""
    return transforms.Compose([
        transforms.ToTensor(),  # Convert PIL Image to tensor first
        transforms.RandomRotation(
            degrees=15,
            fill=0
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=(-5, 5),
            fill=0
        ),
        transforms.RandomPerspective(
            distortion_scale=0.2,
            p=0.5,
            fill=0
        ),
        transforms.RandomErasing(p=0.1),
        transforms.Normalize(
            mean=(0.1307,),
            std=(0.3081,)
        )
    ])

def get_test_transforms():
    """Get test/validation data transformations."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.1307,),
            std=(0.3081,)
        )
    ])

def train_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory for plots
    os.makedirs('training_plots', exist_ok=True)
    
    # Get transforms
    train_transform = get_train_transforms()
    test_transform = get_test_transforms()
    
    # Load MNIST dataset with transforms
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=get_train_transforms()  # Apply transforms directly
    )
    
    # Split into train and validation
    train_size = 50000
    val_size = 10000
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size]
    )
    
    # Override val_dataset transform
    val_dataset.dataset.transform = get_test_transforms()
    
    # Load test dataset
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        transform=get_test_transforms()
    )
    
    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model
    model = MNIST_DNN().to(device)
    print(f"\nTotal parameters: {model.count_parameters():,}")
    criterion = nn.NLLLoss()
    
    # Modified optimizer settings
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=0.001,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Modified scheduler for better learning rate control
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=20,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=1000
    )
    
    # Lists to store metrics
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Training loop
    epochs = 20
    best_accuracy = 0
    
    print("\nTraining Progress:")
    print("-" * 80)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Training with tqdm progress bar
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for data, target in train_pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{running_loss/(train_pbar.n+1):.4f}',
                'acc': f'{100.*correct_train/total_train:.2f}%'
            })
            
        train_loss = running_loss/len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Valid]')
        with torch.no_grad():
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': f'{val_loss/(val_pbar.n+1):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        val_loss = val_loss/len(val_loader)
        accuracy = 100 * correct / total
        
        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_accuracy)
        val_losses.append(val_loss)
        val_accs.append(accuracy)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.2f}%")
        print("-" * 80)
        
        # Update learning rate
        scheduler.step()
        
        # Plot metrics and save model if best accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f'mnist_model_{timestamp}_{accuracy:.2f}.pth'
            
            # Delete previous checkpoints and plots
            for pattern in ['mnist_model_*.pth', 
                          'training_plots/metrics_epoch_*.png',
                          'training_plots/confusion_matrix_*.png']:
                for file in glob.glob(pattern):
                    try:
                        os.remove(file)
                    except:
                        print(f"Could not delete {file}")
            
            # Save model and plots
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'train_losses': train_losses,
                'train_accs': train_accs,
                'val_losses': val_losses,
                'val_accs': val_accs,
            }, model_path)
            print(f'Saved best model checkpoint: {model_path}')
            
        if accuracy > 99.4:
            print(f"\nReached target accuracy of {accuracy:.2f}% (>99.4%). Stopping training.")
            break
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    
    # Final metrics plot with basic matplotlib styling
    plt.figure(figsize=(15, 6))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.title('Loss over epochs', pad=15, fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.legend(fontsize=10)
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.plot(train_accs, label='Training Accuracy', color='blue', linewidth=2)
    plt.plot(val_accs, label='Validation Accuracy', color='red', linewidth=2)
    plt.title('Accuracy over epochs', pad=15, fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Accuracy (%)', fontsize=10)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('training_plots/final_training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Final confusion matrix with enhanced visualization
    conf_matrix = torch.zeros(10, 10)
    for t, p in zip(all_targets, all_preds):
        conf_matrix[t, p] += 1
    
    plt.figure(figsize=(12, 10))
    plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.title('Final Confusion Matrix', pad=15, fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Add numbers to the plot with better formatting
    for i in range(10):
        for j in range(10):
            color = 'white' if conf_matrix[i, j] > conf_matrix.max()/2 else 'black'
            plt.text(j, i, f'{int(conf_matrix[i, j])}',
                    ha="center", va="center", color=color, fontsize=10)
    
    plt.savefig('training_plots/final_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return model

if __name__ == "__main__":
    train_model() 