'''
evaluate.py

a set of functions for evaluating a PyTorch model
based on Raymond Cheng
https://github.com/itsuncheng/fake_news_classification

& 

PPHA 30255 homework 4

'''

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Training Function

def train_model(model,
                optimizer,
                train_loader,
                valid_loader,
                criterion = nn.CrossEntropyLoss(),
                num_epochs = 2,
                file_path = './data',
                best_valid_loss = float("Inf"),
                device='cpu'):
    '''
    Runs training and evaluation on validation data loop over specified number of epochs.
    
    Inputs:
        - model: a PyTorch model object. In this case, the one from lstm.py.
        - optimizer: a PyTorch Optimizer to be used.
        - train_loader: the iterator with training data.
        - valid_loader: the iterator with validation data.
        - criterion: a PyTorch loss function instance.
        - num_epochs: int, how many epochs to train over.
        - file_path: string, where to save model specs.
        - best_valid_loss: float, defaults to infinity to start training from scratch, 
            but if continuing from a prior run and wish to only save better outcomes,
            you can pass the last best model's loss.
        - device: string, 'cpu' or 'cuda' if running on google colab.
    
    Returns: None.
    
    Other Effects:
        - Saves the best model's state dictionary to designated
        file path as 'model.pt'
        - Saves loss history to designated file path as '/metrics.pt'
    '''
    
    eval_every = len(train_loader) // 2
    
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        for (labels, (text, text_len)), _ in train_loader:           
            labels = labels.to(device)
            text = text.to(device)
            text_len = text_len.to(device)
            output = model(text, text_len)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():                    
                  # validation loop
                  for (labels, (text, text_len)), _ in valid_loader:
                        labels = labels.to(device)
                        text = text.to(device)
                        text_len = text_len.to(device)
                        output = model(text, text_len)
                        loss = criterion(output, labels)
                        valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_valid_loss))
                
                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/model.pt', model, optimizer, best_valid_loss)
                    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    
    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')
    
    return None



# Save and Load Functions

def save_checkpoint(save_path, model, optimizer, valid_loss):
    '''
    Used in train_model() to save the current best model.
    
    Inputs:
        - save_path: string, where to save model specs.
        - model: a PyTorch model object.
        - optimizer: a PyTorch Optimizer to be used.
        - train_loader: the iterator with training data.
        - valid_loss: float, current model loss.
        
    Returns: None.
    
    Other Effects:
        - Saves the best model's state dictionary to designated
        file path as 'model.pt'
    '''

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, optimizer, device='cpu'):
    '''
    Used in evaluate() to load the current best model.
    
    Inputs:
        - load_path: string, where to save model specs.
        - model: a PyTorch model object.
        - optimizer: a PyTorch Optimizer to be used.
        - device: string, 'cpu' or 'cuda' if using google colab.
        
    Returns: float, the models last validation loss.
    
    Other Effects:
        - Loads the saved state at load_path
        into the current model object
    '''
    if load_path==None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    '''
    Used in train_model() to save the current best model.
    
    Inputs:
        - save_path: string, where to save model specs.
        - train_loss_list: list of float, current model training losses.
        - valid_loss_list: list of float, current model validation losses.
        - global_steps_list: list of loss function calculation steps.
        
    Returns: None.
    
    Other Effects:
        - Saves the best model's loss history designated
        file path as 'metrics.pt'
    '''
    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path, device='cpu'):
    '''
    Used in evaluate() to load the current best model.
    
    Inputs:
        - load_path: string, where find model specs.
        - device: string, 'cpu' or 'cuda' if using google colab.
    Returns: tuple (train_loss_list, valid_loss_list, global_steps_list)
        - train_loss_list: list of float, current model training losses.
        - valid_loss_list: list of float, current model validation losses.
        - global_steps_list: list of loss function calculation steps.
    '''
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    return (state_dict['train_loss_list'], 
            state_dict['valid_loss_list'], 
            state_dict['global_steps_list'])

# Evaluation Function

def evaluate(model, test_loader, device='cpu'):
    '''
    Accepts the current best model and evaluates
    the test dataset. Printing test accuracy and
    an sklearn confusion matrix report.
    
    Inputs:
    - model: PyTorch model object, the current best model.
    - test_loader: an iterator with test data.
    - device: string, 'cpu' or 'cuda' if using google colab.
    
    Returns: None.
    
    Other Effects:
        Prints test accuracy.
        Prints an Accuracy / F1 Report (sklearn)
        Prints a Confusion Matrix (sklearn & matplotlib)
    '''
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for (labels, (text, text_len)), _ in test_loader:           
            labels = labels.to(device)
            text = text.to(device)
            text_len = text_len.to(device)
            output = model(text, text_len)            
            output = torch.max(output, axis=1).indices
            y_pred.extend(output.tolist())
            y_true.extend(labels.tolist())
    
    acc = 0.0
    for i, y in enumerate(y_pred):
        if y == y_true[i]:
            acc += 1.0
    
    acc = acc / len(y_pred)
    
    
    print('Test Accuracy: ', acc)
    
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))
    
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['HIGH','LOW'])
    ax.yaxis.set_ticklabels(['HIGH','LOW'])
    
    return None
