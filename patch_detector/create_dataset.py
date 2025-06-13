import torch
import torchvision.models as models
import numpy as np
from PIL import Image
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_name',
                        type=str,
                        default='detector_dataset')
    parser.add_argument('--num_images',
                        type=int,
                        default=3000)
    parser.add_argument('--use_gpu', 
                        action='store_true')
    parser.add_argument('--gpu_num',
                        type=int,
                        default=0)
    args = parser.parse_args()
    return args

def main(args):
    model = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1').eval().to(args.device)
    
    attacked_stack = None
    benign_stack = None 
    for i in range(args.num_images):
        if (i%50) == 0: print(f'Total CSV Lines: {i}')
        
        try:
            a = Image.open(f'<INSERT PATH TO ATTACKED IMAGES HERE>').convert('RGB')
            a = to_tensor(a).to(DEVICE).unsqueeze(0)
            b = Image.open(f'<INSERT PATH TO BENIGN IMAGES HERE>').convert('RGB')
            b = to_tensor(b).to(DEVICE).unsqueeze(0)
        except:
            continue
        
        with torch.no_grad():
            a_out = model(normalize(a))
            b_out = model(normalize(b))
          
        a_out = a_out.detach().cpu().numpy()
        b_out = b_out.detach().cpu().numpy()
        
        a_out = np.insert(a_out, 0, 1)
        b_out = np.insert(b_out, 0, 0)
        
        if type(attacked_stack) == type(None):
            attacked_stack = a_out
            benign_stack = b_out
        elif attacked_stack.shape[0] == 1001 and len(attacked_stack.shape) == 1:
            attacked_stack = np.stack((attacked_stack, a_out))
            benign_stack = np.stack((benign_stack, b_out))
        else:
            attacked_stack = np.concatenate((attacked_stack, np.expand_dims(a_out, axis=0)))
            benign_stack = np.concatenate((benign_stack, np.expand_dims(b_out, axis=0)))
            
    dataset = np.concatenate((attacked_stack, benign_stack)) 
    csv = open(f'{args.csv_name}.csv', 'ab')
    np.savetxt(csv, dataset, delimiter=',')
    csv.close()

if __name__ == '__main__':
    args = parse_args()
    args.device = args.device = f'cuda:{args.gpu_num}' if args.use_gpu else 'cpu'
    
    main(args)