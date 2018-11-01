import argparse
import json
from collections import OrderedDict
import os

import torch
from torch import nn, optim
import torchvision
from torchvision import transforms, datasets, models

parser = argparse.ArgumentParser(description='Model Training')
parser.add_argument('data_dir', type=str, help='data directory')
parser.add_argument('--save_dir', type=str, help='checkpoint saving directory', default='./checkpoint')
parser.add_argument('--arch', type=str, help='model architecture', default='vgg19_bn')
parser.add_argument('--learning_rate', type=float, help='learning rate', default=0.001)
parser.add_argument('--first_hidden_units', type=int, help='first hidden units', default=4096)
parser.add_argument('--second_hidden_units', type=int, help='second hidden units', default=512)
parser.add_argument('--epochs', type=int, help='epochs', default=6)
parser.add_argument('--gpu', action='store_true', help='use gpu')

args = parser.parse_args()

# Data directory
train_dir = args.data_dir + '/train'
valid_dir = args.data_dir + '/valid'
test_dir = args.data_dir + '/test'

# Define Transformation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Training data transforms: introducing randomness
data_transforms_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Validation and test data transfom: no need for randomness
data_transforms_valid_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Load the datasets with ImageFolder
image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(train_dir, transform=data_transforms_train)
image_datasets['valid'] = datasets.ImageFolder(valid_dir, transform=data_transforms_valid_test)
image_datasets['test'] = datasets.ImageFolder(test_dir, transform=data_transforms_valid_test)

# Using the image datasets and the trainforms, define the dataloaders
dataloaders = {}
dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=16, shuffle=True)
dataloaders['valid'] = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=16, shuffle=True)
dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=16, shuffle=True)

# Label Mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Build the model
# Use VGG for this training
architecture = args.arch.lower()
if architecture == 'vgg11':
    model = models.vgg11(pretrained=True)
elif architecture == 'vgg11_bn':
    model = models.vgg11_bn(pretrained=True)
elif architecture == 'vgg13':
    model = models.vgg13(pretrained=True)
elif architecture == 'vgg13_bn':
    model = models.vgg13_bn(pretrained=True)
elif architecture == 'vgg16':
    model = models.vgg16(pretrained=True)
elif architecture == 'vgg16_bn':
    model = models.vgg16_bn(pretrained=True)
elif architecture == 'vgg19':
    model = models.vgg19(pretrained=True)
elif architecture == 'vgg19_bn':
    model = models.vgg19_bn(pretrained=True)
else:
    raise ValueError('--arch must be assigned a vgg model name')
  
# Freeze parameters
for param in model.parameters():
    param.requires_grad = False

# Replace the classifier
num_input = model.classifier[0].in_features
hidden_units = [args.first_hidden_units, args.second_hidden_units]
num_output = len(cat_to_name)
model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(num_input, hidden_units[0])),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(0.5)),
                          ('fc2', nn.Linear(hidden_units[0], hidden_units[1])),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout(0.5)),
                          ('fc3', nn.Linear(hidden_units[1], num_output)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
# Print model info
print('The model is {}'.format(model.__class__.__name__))
print('The classifier structure: ')
print(model)

# Define loss and optimizer functions
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

# Accuracy function: validation or test
def check_accuracy_loss(model, dataloaders, phase, criterion, gpu=False):
    loss = 0
    accuracy = 0

    if gpu:
        model.to('cuda')

    for images, labels in dataloaders[phase]:
        if gpu:
            images, labels = images.to('cuda'), labels.to('cuda')
        outputs = model.forward(images)
        loss += criterion(outputs, labels).item()
        ps = torch.exp(outputs)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    accuracy = accuracy / len(dataloaders[phase])
    loss = loss / len(dataloaders[phase])
    return accuracy, loss

def perform_learning(model, dataloaders, criterion, optimizer, epochs=3, print_every=40, gpu=False):

    step = 0
    # change to cuda
    if gpu:
        model.to('cuda')

    for epoch in range(1, epochs+1):
        running_loss = 0
        scheduler.step()
        for ii, (inputs, labels) in enumerate(dataloaders['train']):
            step += 1
            if gpu:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % print_every == 0:
                # Check for validation
                model.eval()
                with torch.no_grad():
                    accuracy, valid_loss =  check_accuracy_loss(model, dataloaders, 'valid', criterion, gpu)

                print('Epoch: {}/{}'.format(epoch, epochs),
                      'Training Loss: {:.3f}..'.format(running_loss/print_every),
                      'Valid Loss: {:.3f}..'.format(valid_loss),
                      'Valid Accuracy: {:.3f}'.format(accuracy))

                running_loss = 0

                model.train()

# Do validation on the test set
def test(model, gpu):
    model.eval()
    with torch.no_grad():
        accuracy, test_loss =  check_accuracy_loss(model, dataloaders, 'test', criterion, gpu)
        print('Test Loss: {:.3f}..'.format(test_loss),
              'Test Accuracy: {:.3f}'.format(accuracy))

# Do the learning
use_gpu = args.gpu
epochs = args.epochs
perform_learning(model, dataloaders, criterion, optimizer, epochs=epochs, print_every=50, gpu=use_gpu)
print('The model has been trained. Now checking with test datasets')
test(model, use_gpu)

# Check if directory exists
# If not, create one
save_dir = args.save_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
# Save the checkpoint
model.class_to_idx = image_datasets['train'].class_to_idx
torch.save({
    'transfer_model': architecture,
    'classifier': {
        'number_of_hidden_layer': 2,
        'input_layer': num_input,
        'hidden_layer': hidden_units,
        'output_output_layer': num_output
    },
    'state_dict': model.state_dict(),
    'class_to_idx': model.class_to_idx,
    'optimizer.state_dict': optimizer.state_dict,
    'epochs': epochs
}, save_dir + '/checkpoint.pth')
