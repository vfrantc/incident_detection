import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from warmup_scheduler import GradualWarmupScheduler

from dataset import ConsistentVideoTransform
from dataset import RGBFrameDataset
from model import VideoClassifier

def main(args):
    # Initialize the dataset and dataloader
    train_transform = ConsistentVideoTransform(
        resize_size=(350, 350),
        crop_size=(320, 320),
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )

    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
    ])

    train_dataset = RGBFrameDataset(root=args.data_root, n_frames=45, transform=train_transform, is_train=True)
    test_dataset = RGBFrameDataset(root=args.data_root, n_frames=45, transform=transform, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=train_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=test_dataset.collate_fn)

    # Initialize the model
    model = VideoClassifier(num_classes=args.num_classes, lstm_hidden_size=512, lstm_num_layers=2)
    model = model.cuda()  # move model to GPU if available

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.start_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.000001)

    # Learning rate scheduler with warmup and cosine annealing
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs - args.warmup_epochs, eta_min=args.end_lr)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warmup_epochs, after_scheduler=scheduler_cosine)

    # Training and validation
    best_accuracy = 0.0

    for epoch in range(args.num_epochs):
        model.train()  # set the model to training mode
        running_loss = 0.0
        for i, (videos, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            videos = videos.cuda()  # move data to GPU if available
            labels = labels.cuda()  # move data to GPU if available

            # Forward pass
            outputs = model(videos)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)  # Gradient clipping
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Loss: {epoch_loss}')

        # Validate the model
        model.eval()  # set the model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (videos, labels) in tqdm(enumerate(test_loader), total=len(test_loader)):
                videos = videos.cuda()  # move data to GPU if available
                labels = labels.cuda()  # move data to GPU if available

                outputs = model(videos)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        validation_accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Validation Accuracy: {validation_accuracy}%')

        # Save the best model
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print('Best model saved with accuracy:', best_accuracy)

        # Step the scheduler
        scheduler.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a video classification model.')
    parser.add_argument('--data_root', type=str, default='data', help='root directory of the dataset')
    parser.add_argument('--num_epochs', type=int, default=25, help='number of epochs to train')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='number of warmup epochs')
    parser.add_argument('--start_lr', type=float, default=0.0001, help='starting learning rate')
    parser.add_argument('--end_lr', type=float, default=0.000001, help='ending learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')
    parser.add_argument('--num_classes', type=int, default=2, help='number of output classes')

    args = parser.parse_args()
    main(args)
