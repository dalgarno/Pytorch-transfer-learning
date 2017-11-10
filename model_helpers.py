import time
import os
import csv
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn.functional as F

def train_model(model, criterion, optimizer, scheduler, datasets, datset_sizes, use_gpu=True, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in datasets[phase]:
                # get the inputs
                inputs, labels = data['image'], data['label']

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Test the model
def test_model(model, criterion, optimizer, scheduler, datasets, dataset_sizes, use_gpu=True, num_epochs=1):
    model.train(False)  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for data in datasets['test']:
        # get the inputs
        inputs, labels = data['image'], data['label']

        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.data[0]
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_sizes['test']
    epoch_acc = running_corrects / dataset_sizes['test']

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        'test', epoch_loss, epoch_acc))
    

def get_model_predictions(model, optimizer, labels, submission_csv, 
                          submission_dataloader, submission_image_names, 
                          use_gpu=True, num_epochs=1):
    model.train(False)  # Set model to evaluate mode
    
    if os.path.exists(submission_csv):
        os.remove(submission_csv)
    
    # Write predictions to csv
    submission_labels = ['id'] + labels
    with open(submission_csv, 'w') as f:
        writer = csv.writer(f, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(submission_labels)
    
        # Iterate over data.
        for i, data in enumerate(submission_dataloader):

            # get the inputs, no labels
            inputs, _ = data['image'], data['label']
            # wrap them in Variable
            if use_gpu:
                inputs = Variable(inputs.cuda())
            else:
                inputs = Variable(inputs)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            rounded_outputs = np.around(F.softmax(outputs).cpu().data.numpy()[0], 17)
            each_row = [submission_image_names[i]] + rounded_outputs.tolist()
            writer.writerow(each_row)
            
    print('done!')