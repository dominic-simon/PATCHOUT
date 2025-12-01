from csv_dataset import CSVDataset
from patch_detector import PatchDetector as Detector

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_model_name',
                        type=str,
                        default='detector')
    parser.add_argument('--test_model_name',
                        type=str,
                        default='detector')
    parser.add_argument('--train_dataset_name',
                        type=str,
                        default='detector')
    parser.add_argument('--test_dataset_name',
                        type=str,
                        default='detector')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64)
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001)
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9)
    parser.add_argument('--num_epochs',
                        type=int,
                        default=3)
    parser.add_argument('--do_train',
                        action='store_true')
    parser.add_argument('--do_test',
                        action='store_true')
    parser.add_argument('--use_gpu', 
                        action='store_true')
    parser.add_argument('--gpu_num',
                        type=int,
                        default=0)
    args = parser.parse_args()
    return args

def train(args): 
    train_df = pd.read_csv(f'{args.train_dataset_name}.csv', header=None)
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    trainset = CSVDataset(train_df)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    model = Detector().float().train().to(args.device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    for epoch in range(args.num_epochs):
      running_loss = 0.0
      for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].float().to(args.device), data[1].type(torch.LongTensor).to(args.device)
    
        optimizer.zero_grad()
        outputs = model(inputs)
    
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 10 == 9:
        	print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
        	running_loss = 0.0

    torch.save(model.state_dict(), f'{args.save_model_name}.pt') 
    
def test(args):
    test_df = pd.read_csv(f'{args.test_dataset_name}.csv', header=None)
    test_df = test_df.sample(frac=1).reset_index(drop=True)
    testset = CSVDataset(test_df)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=4)
    
    model.load_state_dict(torch.load(f'{args.test_model_name}.pt'))
    model = model.eval().to(args.device)
    
    acc = 0
    a_acc = 0
    a_count = 0
    b_acc = 0
    b_count = 0
    count = 0
    for i, data in tqdm(enumerate(testloader, 0)):
        inputs, labels = data[0].float().to(args.device), data[1].item()
        output = model(inputs).argmax(dim=1).item()
            
        if labels == 1:
          a_count += 1
          if output == labels:
              acc += 1
              a_acc +=1
        elif labels == 0:
          b_count += 1
          if output == labels:
              acc += 1
              b_acc +=1 
        count += 1
        
    print(f'Adv Accuracy: {a_acc/a_count*100}%')
    print(f'Benign Accuracy: {b_acc/b_count*100}%') 
    print(f'Total Accuracy: {acc/count*100}%')

if __name__ == '__main__':
    args = parse_args()
    args.device = args.device = f'cuda:{args.gpu_num}' if args.use_gpu else 'cpu'
    
    if args.do_train:
        train(args)
    if args.do_test:
        test(args)
