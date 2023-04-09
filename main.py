import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image

import transforms as ext_transforms
from models.enet import ENet
from models.fcn import make_fcn
from train import Train
from test import Test
from metric.iou import IoU
from args import get_arguments
from data.utils import enet_weighing, median_freq_balancing
import utils

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Get the arguments
args = get_arguments()

device = torch.device(args.device)


def load_dataset(dataset):
    print("\nLoading dataset...\n")

    print("Selected dataset:", args.dataset)
    print("Dataset directory:", args.dataset_dir)
    print("Save directory:", args.save_dir)

    image_transform = transforms.Compose(
        [transforms.Resize((args.height, args.width)),
         transforms.ToTensor()])

    label_transform = transforms.Compose([
        transforms.Resize((args.height, args.width), Image.NEAREST),
        ext_transforms.PILToLongTensor(),
        ext_transforms.ToOnehotGaussianBlur(kernel_size=7, num_classes=dataset.num_classes)
    ])

    # Get selected dataset
    # Load the training set as tensors
    whole_train_set = dataset(
        args.dataset_dir,
        transform=image_transform,
        label_transform=label_transform)

    # change the split ratio and do not use the original validation set,
    # because I want to use it as the test set for conformal prediction
    train_set, val_set, cal_set, _ = data.random_split(whole_train_set, [1800, 200, 600, len(whole_train_set) - 2600],
                                                       generator=torch.Generator().manual_seed(0))

    train_loader = data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)
    val_loader = data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)
    cal_loader = data.DataLoader(
        cal_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    # Load the test set as tensors
    test_set = dataset(
        args.dataset_dir,
        mode='test',
        transform=image_transform,
        label_transform=label_transform)
    test_loader = data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    # Get encoding between pixel valus in label images and RGB colors
    class_encoding = whole_train_set.color_encoding

    # Remove the road_marking class from the CamVid dataset as it's merged
    # with the road class
    if args.dataset.lower() == 'camvid':
        del class_encoding['road_marking']

    # Get number of classes to predict
    num_classes = len(class_encoding)

    # Print information for debugging
    print("Number of classes to predict:", num_classes)
    print("Train dataset size:", len(train_set))
    print("Validation dataset size:", len(val_set))

    # Get a batch of samples to display
    if args.mode.lower() == 'test':
        images, labels, regression_labels = iter(test_loader).next()
    else:
        images, labels, regression_labels = iter(train_loader).next()
    print("Image size:", images.size())
    print("Label size:", labels.size())
    print("Class-color encoding:", class_encoding)

    # Show a batch of samples and labels
    if args.imshow_batch:
        print("Close the figure window to continue...")
        label_to_rgb = transforms.Compose([
            ext_transforms.LongTensorToRGBPIL(class_encoding),
            transforms.ToTensor()
        ])
        color_labels = utils.batch_transform(labels, label_to_rgb)
        utils.imshow_batch(images, color_labels)

    # Get class weights from the selected weighing technique
    print("\nWeighing technique:", args.weighing)
    print("Computing class weights...")
    print("(this can take a while depending on the dataset size)")
    class_weights = 0
    if args.weighing.lower() == 'enet':
        class_weights = enet_weighing(train_loader, num_classes)
    elif args.weighing.lower() == 'mfb':
        class_weights = median_freq_balancing(train_loader, num_classes)
    else:
        class_weights = None

    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float().to(device)
        # Set the weight of the unlabeled class to 0
        if args.ignore_unlabeled:
            ignore_index = list(class_encoding).index('unlabeled')
            class_weights[ignore_index] = 0

    print("Class weights:", class_weights)

    return (train_loader, val_loader, cal_loader,
            test_loader), class_weights, class_encoding


def train(model_name, train_loader, val_loader, class_weights, class_encoding, regression=False):
    print("\nTraining...\n")

    num_classes = len(class_encoding)

    # Intialize ENet
    if model_name.lower() == 'enet':
        model = ENet(num_classes).to(device)
    elif model_name.lower() == 'fcn':
        model = make_fcn(pretrained=True, img_shape=(args.height, args.width), num_classes=num_classes)
        model = model.to(device)
    else:
        raise ValueError
    # Check if the network architecture is correct
    print(model)

    # We are going to use the CrossEntropyLoss loss function as it's most
    # frequentely used in classification problems with multiple classes which
    # fits the problem. This criterion  combines LogSoftMax and NLLLoss.
    if regression:
        class WeightedMSE(nn.Module):
            def __init__(self, weight=None):
                super(WeightedMSE, self).__init__()
                if weight is not None:
                    self.weight = weight / weight.sum()
                else:
                    self.weight = None
            def forward(self, inputs, targets):
                if self.weight is not None:
                    mse = ((inputs - targets) ** 2).mean(dim=(0, 2, 3))  # (num_classes,)
                    mse = (self.weight * mse).sum()
                else:
                    mse = ((inputs - targets) ** 2).mean()
                return mse
        criterion = WeightedMSE(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ENet authors used Adam as the optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay)

    # Learning rate decay scheduler
    lr_updater = lr_scheduler.StepLR(optimizer, args.lr_decay_epochs,
                                     args.lr_decay)

    # Evaluation metric
    if args.ignore_unlabeled:
        ignore_index = list(class_encoding).index('unlabeled')
    else:
        ignore_index = None
    metric = IoU(num_classes, ignore_index=ignore_index, is_regression=regression)

    # Optionally resume from a checkpoint
    if args.resume:
        model, optimizer, start_epoch, best_miou = utils.load_checkpoint(
            model, optimizer, args.save_dir, args.name)
        print("Resuming from model: Start epoch = {0} "
              "| Best mean IoU = {1:.4f}".format(start_epoch, best_miou))
    else:
        start_epoch = 0
        best_miou = 0

    # Start Training
    print()
    train = Train(model, train_loader, optimizer, criterion, regression, metric, device)
    val = Test(model, val_loader, criterion, regression, metric, device)
    for epoch in tqdm(range(start_epoch, args.epochs)):
        print(">>>> [Epoch: {0:d}] Training".format(epoch))

        epoch_loss, (iou, miou) = train.run_epoch(args.print_step)
        lr_updater.step()

        print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
              format(epoch, epoch_loss, miou))

        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(">>>> [Epoch: {0:d}] Validation".format(epoch))

            loss, (iou, miou) = val.run_epoch(args.print_step)

            print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
                  format(epoch, loss, miou))

            # Print per class IoU on last epoch or if best iou
            if epoch + 1 == args.epochs or miou > best_miou:
                for key, class_iou in zip(class_encoding.keys(), iou):
                    print("{0}: {1:.4f}".format(key, class_iou))

            # Save the model if it's the best thus far
            if miou > best_miou:
                print("\nBest model thus far. Saving...\n")
                best_miou = miou
                utils.save_checkpoint(model, optimizer, epoch + 1, best_miou,
                                      args)

    return model


def test(model, test_loader, class_weights, class_encoding, regression=False):
    print("\nTesting...\n")

    num_classes = len(class_encoding)

    if regression:
        # Because we modify the segmentation problem into a regression task, we need to use this MSE loss.
        class WeightedMSE(nn.Module):
            def __init__(self, weight=None):
                super(WeightedMSE, self).__init__()
                if weight is not None:
                    self.weight = weight / weight.sum()
                else:
                    self.weight = None

            def forward(self, inputs, targets):
                if self.weight is not None:
                    mse = ((inputs - targets) ** 2).mean(dim=(0, 2, 3))  # (num_classes,)
                    mse = (self.weight * mse).sum()
                else:
                    mse = ((inputs - targets) ** 2).mean()
                return mse

        criterion = WeightedMSE(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Evaluation metric
    if args.ignore_unlabeled:
        ignore_index = list(class_encoding).index('unlabeled')
    else:
        ignore_index = None
    metric = IoU(num_classes, ignore_index=ignore_index, is_regression=regression)

    # Test the trained model on the test set
    test = Test(model, test_loader, criterion, regression, metric, device)

    print(">>>> Running test dataset")

    loss, (iou, miou) = test.run_epoch(args.print_step)
    class_iou = dict(zip(class_encoding.keys(), iou))

    print(">>>> Avg. loss: {0:.4f} | Mean IoU: {1:.4f}".format(loss, miou))

    # Print per class IoU
    for key, class_iou in zip(class_encoding.keys(), iou):
        print("{0}: {1:.4f}".format(key, class_iou))

    # Show a batch of samples and labels
    if args.imshow_batch:
        print("A batch of predictions from the test set...")
        images, _, _ = iter(test_loader).next()
        predict(model, images, class_encoding, regression)


def predict(model, images, class_encoding, is_regression):
    images = images.to(device)

    # Make predictions!
    model.eval()
    with torch.no_grad():
        predictions = model(images)

    if is_regression:
        predictions = torch.exp(-torch.exp(predictions))

    # Predictions is one-hot encoded with "num_classes" channels.
    # Convert it to a single int using the indices where the maximum (1) occurs
    _, predictions = torch.max(predictions.data, 1)

    label_to_rgb = transforms.Compose([
        ext_transforms.LongTensorToRGBPIL(class_encoding),
        transforms.ToTensor()
    ])
    color_predictions = utils.batch_transform(predictions.cpu(), label_to_rgb)
    utils.imshow_batch(images.data.cpu(), color_predictions)


# Run only if this module is being run directly
if __name__ == '__main__':

    # Fail fast if the dataset directory doesn't exist
    assert os.path.isdir(
        args.dataset_dir), "The directory \"{0}\" doesn't exist.".format(
            args.dataset_dir)

    # Fail fast if the saving directory doesn't exist
    assert os.path.isdir(
        args.save_dir), "The directory \"{0}\" doesn't exist.".format(
            args.save_dir)

    # Import the requested dataset
    if args.dataset.lower() == 'camvid':
        from data import CamVid as dataset
    elif args.dataset.lower() == 'cityscapes':
        from data import Cityscapes as dataset
    else:
        # Should never happen...but just in case it does
        raise RuntimeError("\"{0}\" is not a supported dataset.".format(
            args.dataset))

    loaders, w_class, class_encoding = load_dataset(dataset)
    train_loader, val_loader, cal_loader, test_loader = loaders

    if args.mode.lower() in {'train', 'full'}:
        model = train(args.model_name, train_loader, val_loader, w_class, class_encoding, args.regression)

    if args.mode.lower() in {'test', 'full'}:
        if args.mode.lower() == 'test':
            # Intialize a new ENet model
            num_classes = len(class_encoding)
            if args.model_name.lower() == 'enet':
                model = ENet(num_classes).to(device)
            elif args.model_name.lower() == 'fcn':
                model = make_fcn(pretrained=True, img_shape=(args.height, args.width), num_classes=num_classes)
                model = model.to(device)
            else:
                raise ValueError

        # Initialize a optimizer just so we can retrieve the model from the
        # checkpoint
        optimizer = optim.Adam(model.parameters())

        # Load the previoulsy saved model state to the ENet model
        model = utils.load_checkpoint(model, optimizer, args.save_dir,
                                      args.name)[0]

        if args.mode.lower() == 'test':
            print(model)

        test(model, test_loader, w_class, class_encoding)
