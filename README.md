# PATCHOUT: Adversarial Patch Detection and Localization using Semantic Consistency

This is the repository for [PATCHOUT: Adversarial Patch Detection and Localization using Semantic Consistency](https://link.springer.com/article/10.1007/s11063-025-11775-5), which is published in the Springer journal Neural Processing Letters. 

## PATCHOUT
__Patch Detector__

A few examples of trained patch detectors are provided in ```/patch_detector/```. The script to create a new dataset for training custom detectors is provided in ```/patch_detector/create_dataset.py```. The script to train a new patch detector is given in ```/patch_detector/train.py```. Refer to Table 2 (page 8) of the paper for parameter selection. 

__Patch Removal__

The checkpoints for the Segment Anything Model (SAM) can be downloaded from the Github repository [here](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints). There are newer version of SAM available. If you wish to use these version, you might need to update the code to reflect updated classes and function calls. 

The original Github repository for the inpainting approach used can be found [here](https://github.com/rlct1/gin), and the inpainting model download is located under the _Testing_ header. If this link disappears, or if you just feel like, the inpainting method can be switched for another inpainting method. Of course, this will require some code rewriting, but it shouldn't take too much work. The following directories and files are necessary for inpainting: ```/checkpoints/gin/```, ```/data/```, ```/datasets/ade20k/test```, ```/models/```, ```/options/```, ```/results/test/```, ```/util/```, and ```inpaint.py```. The listed directories contain files necessary for inpainting and ```inpaint.py``` is the file that enacts the inpainting. 

The PATCHOUT framework itself is contained in the ```PATCHOUT.py``` file. To instantiate the framework in code, you need to pass a computer vision model and the hardware device to run the framework on (```defense = Patchout(model=<your_model_here>, device=<'cuda' or 'cpu'>)```). To defend an image, you need to pass the image and the attacked image class as predicted by the model to the ```defend()``` method (```defended_image = defense.defend(image=<current_image>, class_num=<top1_model_prediction_int>)```).

## Bugs and Questions
If you come across any bugs or issues with the code, please feel free to email Dominic Simon at **dominic.simon@ufl.edu** or open an issue on this repository.

## Citation
If you use our code or find the work helpful, please consider citing our work:

**Bibtex**
```bibtex
@article{simon2025patchout,
  title     = {PATCHOUT: Adversarial Patch Detection and Localization using Semantic Consistency},
  author    = {Simon, Dominic and Jha, Sumit and Ewetz, Rickard},
  journal   = {Neural Processing Letters},
  publisher = {Springer Nature},
  volume    = {57},
  year      = {2025},
  month     = {6},
  date      = {2025-06-01},
  issn      = {1573-773X},
  doi       = {10.1007/s11063-025-11775-5},
  url       = {https://doi.org/10.1007/s11063-025-11775-5},
}
```

**RIS**
```ris
TY  - JOUR
AU  - Simon, Dominic
AU  - Jha, Sumit
AU  - Ewetz, Rickard
PY  - 2025
DA  - 2025/06/01
TI  - PATCHOUT: Adversarial Patch Detection and Localization using Semantic Consistency
JO  - Neural Processing Letters
SP  - 55
VL  - 57
IS  - 3
AB  - Computer vision systems are actively deployed in safety-critical applications such as autonomous vehicles. Real-world adversarial patches are capable of compromising the artificial intelligence (AI) systems with catastrophic outcomes. Existing defenses against patch attacks are based on identifying neurons, features, or gradients of high intensity. However, these defenses are vulnerable to weaker attacks that have less obvious attack signatures. In this paper, we propose the PATCHOUT framework that detects and locates adversarial patches using semantic consistency. Within patch detection, the key insight is that the top class predictions for an entity are semantically consistent for benign images, whereas they are inconsistent for attacked images. Within patch localization, it is observed that patches are semantically consistent with a coarse grained segmentation of the image. This allows the PATCHOUT framework to detect and remove adversarial patches using a class consistency checker as well as image segmentation, attribution analysis, and image restoration techniques. The experimental evaluation demonstrates that PATCHOUT can detect a broad range of adversarial patches with over 90% accuracy. The framework achieves 20% higher accuracy than other defenses. The framework is also evaluated against unseen attacks and adaptive attacks, reducing the success rate of adaptive attacks from 56% to 24%.
SN  - 1573-773X
UR  - https://doi.org/10.1007/s11063-025-11775-5
DO  - 10.1007/s11063-025-11775-5
ID  - Simon2025
ER  -
```
