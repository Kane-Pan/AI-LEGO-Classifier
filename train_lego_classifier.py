import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets, transforms, models

import numpy as np
from PIL import Image
import argparse

'''The images in the dataset are not all square, so this function pads the images into
squares so that parts of the images don't get cropped out when training'''
def pad_to_square(img):
    img = img.convert('RGB')
    width, height = img.size

    # Check if already square
    if width == height:
        return img
    
    # Sample corner pixels then take their average
    corners = [
        img.getpixel((0, 0)),
        img.getpixel((width - 1, 0)),
        img.getpixel((0, height - 1)),
        img.getpixel((width - 1, height - 1))
    ]
    corner_array = np.array(corners)
    average_rgb = corner_array.mean(axis=0)
    avg_colour = tuple(average_rgb.astype(int))
    
    # Create a square of avg_colour using the max dimension of the original image
    max_dim = max(width, height)
    new_img = Image.new('RGB', (max_dim, max_dim), color=avg_colour)

    # Place the original image in the center
    new_img.paste(img, ((max_dim - width) // 2, (max_dim - height) // 2))
    return new_img

'''transform to pad images to square'''
class PadToSquare(object):
    def __call__(self, img):
        return pad_to_square(img)

# Define a TransformSubset so that transformations can be used on a given subset
class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.subset[index]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.subset)

'''Define transformations to be done in preprocessing'''
data_transforms = {
    'train': transforms.Compose([
        PadToSquare(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),

    'val': transforms.Compose([
        PadToSquare(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}

'''Add terminal command-line arguments to continue training where it left off'''
parser = argparse.ArgumentParser(description="Start New or Resume Training")
parser.add_argument('--resume_train', action='store_true',
                    help="Resume training from previous checkpoint.")
parser.add_argument('--resume_epochs', type=int, default=4,
                    help="Number of epochs to resume training to.")
parser.add_argument('--total_epochs', type=int, default=10,
                    help="Total epochs if it is new model.")
args = parser.parse_args()

def train_model(model, dataloaders, criterion, optimizer, scheduler, dataset_sizes, device,
                start_epoch=0, num_epochs=8, best_acc=0.0, best_model_wts=None):

    # Make copy of the most accurate model
    if best_model_wts is None:
        best_model_wts = copy.deepcopy(model.state_dict())

    start_time = time.time()

    # Initialize GradScaler for CUDA if available
    if device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = torch.amp.GradScaler('cpu')

    # Main training loop
    for epoch in range(start_epoch, start_epoch + num_epochs):
        print(f"Epoch {epoch+1} out of {start_epoch + num_epochs}")
        print("--o--o--o--o--o--o--o--o--o--o--o--")

        for mode in ['train', 'val']:
            if mode == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[mode]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # Resets gradients
                optimizer.zero_grad()

                # Use mixed precision if available
                with torch.set_grad_enabled(mode == 'train'):
                    with torch.amp.autocast(device_type='cuda', enabled=(scaler is not None)):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)

                        # Using cross-entropy loss good for image classification
                        loss = criterion(outputs, labels)

                    if mode == 'train':
                        if scaler is not None:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / float(dataset_sizes[mode])
            epoch_acc = running_corrects / float(dataset_sizes[mode])

            print(f"{mode} Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.4f}")

            # Save if new model has better accuracy
            if mode == 'val':
                best_acc = max([epoch_acc, best_acc])
                best_model_wts = copy.deepcopy(model.state_dict())

        scheduler.step()
        print()

    time_elapsed = time.time() - start_time
    print(f"Training finished in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best validation Accuracy so far: {best_acc:.4f}")

    last_epoch = start_epoch + num_epochs - 1
    return model, best_acc, best_model_wts, last_epoch


def main():
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Using device:", device)

    # AI was trained on two datasets with same subclass names of lego ID. Concat them into
    # One big dataset
    renders_path = r"" # Copy file path to renders folder here
    photos_path  = r"" # Copy file path to photos folder here

    renders_dataset = datasets.ImageFolder(root=renders_path, transform=None)
    photos_dataset  = datasets.ImageFolder(root=photos_path, transform=None)

    if renders_dataset.class_to_idx != photos_dataset.class_to_idx:
        raise ValueError("Some Subclasses dont match!")

    full_dataset = ConcatDataset([renders_dataset, photos_dataset])

    dataset_size = len(full_dataset)
    # Give 80% of the dataset for training and 20% for the validation
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    # Apply transformations
    train_dataset = TransformSubset(train_subset, transform=data_transforms['train'])
    val_dataset = TransformSubset(val_subset, transform=data_transforms['val'])

    # Can edit some of these parameters depending on training system
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    class_names = renders_dataset.classes
    num_classes = len(class_names)

    save_destination = "" # Paste file path to where you want to save

    # Using ResNet18 to make it efficient since its pre-trained for identifying images
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    start_epoch = 0
    best_acc = 0.0
    best_model_wts = None

    if args.resume_train:
        print(f"\nResuming training for {args.resume_epochs} more epochs.")

        if os.path.exists(save_destination):

            checkpoint = torch.load(save_destination, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            best_model_wts = checkpoint['best_model_wts']

        else:
            raise FileNotFoundError(f"Checkpoint file doesn't exist at '{save_destination}'.")

        model, best_acc, best_model_wts, last_epoch = train_model(
            model, dataloaders, criterion, optimizer, scheduler, dataset_sizes, device,
            start_epoch=start_epoch, num_epochs=args.resume_epochs, best_acc=best_acc,
            best_model_wts=best_model_wts
        )

        torch.save({
            'epoch': last_epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 'best_acc': best_acc, 'best_model_wts': best_model_wts
        }, save_destination)

        print(f"\nModel saved at epoch {last_epoch} with best_acc={best_acc:.4f}!")

    else:
        model, best_acc, best_model_wts, last_epoch = train_model(
            model, dataloaders, criterion, optimizer, scheduler, dataset_sizes, device,
            num_epochs=args.total_epochs
        )

        torch.save({
            'epoch': last_epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 'best_acc': best_acc, 'best_model_wts': best_model_wts
        }, save_destination)

        print(f"\nModel saved at {save_destination} with best_acc={best_acc:.4f}.")


if __name__ == '__main__':
    main()