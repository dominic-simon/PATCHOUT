import torch
from torchvision.utils import save_image
import torchvision.transforms as transforms

from captum.attr import NoiseTunnel
from captum.attr import GuidedBackprop

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class Patchout():
    def __init__(self,
                 model,
                 device='cuda' if torch.cuda.is_available() else 'cpu'
                 ):
        self.model = model.to(device)
        self.device = device

        self.to_grayscale = transforms.Grayscale()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor()])

        self.gb = GuidedBackprop(model)
        self.gb = NoiseTunnel(self.gb)

        sam = sam_model_registry['vit_h'](checkpoint='../sam_weights.pth').to(device)
        self.sam = SamAutomaticMaskGenerator(model=sam, pred_iou_thresh=0.99)

    def defend(self, image, class_num):
        image_clone = image.detach().clone()
        image_clone = np.transpose(image_clone.detach().cpu().numpy(), (1, 2, 0))
        image_clone = (image_clone * 255).astype(np.uint8)
        masks = self.sam.generate(image_clone)

        attr = self.gb.attribute(self.normalize(image.unsqueeze(0)).to(self.device),
                                 nt_type='smoothgrad',
                                 nt_samples=25,
                                 nt_samples_batch_size=5,
                                 stdevs=float(0.15*torch.max(image)-torch.min(image)) if float(0.15*torch.max(image)-torch.min(image)) > 0 else 0.0,
                                 target=class_num)
        attr = self.to_grayscale(attr).squeeze()

        best_mask = None
        highest_attr = 0
        for mask in masks:
            mask = torch.from_numpy(mask['segmentation']).to(self.device)
            masked_attr = mask * attr
            masked_attr = torch.abs(masked_attr)
            average_attr = torch.sum(masked_attr) / torch.count_nonzero(mask == 1)
            if best_mask is None:
                best_mask = mask
                highest_attr = average_attr
            elif average_attr > highest_attr:
                best_mask = mask
                highest_attr = average_attr

        if best_mask is None:
            return image

        mask_3d = torch.stack([best_mask, best_mask, best_mask], dim=0).type(torch.float32)
        image[image == 1.] = 0.9999
        image[mask_3d == 1] = 1
        save_image(image, './datasets/ade20k/test/AIM_IC_t1_validation_0_with_holes.png')
        save_image(mask_3d, './datasets/ade20k/test/AIM_IC_t1_validation_0_mask.png')

        os.system('python3 ./test_ensemble.py --name gin')

        modified_image = Image.open('./results/test/AIM_IC_t1_validation_0.png').convert('RGB')
        modified_image = self.to_tensor(modified_image).to(self.device)

        return modified_image
