# Joint-Multi-Scale-Tone-Mapping-and-Denoising-for-HDR-Image-Enhancement
A pytorch implementation of the "TFDL" and "DFTL" models in the "Joint Multi-Scale Tone Mapping and Denoising for HDR Image Enhancement" paper.

## Requirements
1. Python 3.7 
2. Pytorch 1.11.0
3. opencv
4. torchvision
5. cuda 10.2
6. numpy
7. matplotlib
8. torch_dct
9. kornia

### Folder structure
```

├── examples # Contains input examples to test the model
├── results
│   └── TFDL # Test results will be saved here by default
├── snapshots # Pre-trained snapshots
│   └── TFDL.pth # Pre-trained TFDL model
├── csrnet.py
├── gaussian_pyramid.py
├── main_test.py # testing code
├── models.py # Our models are defined here
```
### Test: 

cd Joint-Multi-Scale-Tone-Mapping-and-Denoising-for-HDR-Image-Enhancement
```
python main_test.py 
```
The script will process the images in "examples" folder and save the enhanced images to "results" folder.

##  License
The code is made available for academic research purpose only. This project is open sourced under MIT license.

## Bibtex

```
@INPROCEEDINGS{9707563,
 author = {Hu, Litao and Chen, Huaijin and Allebach, Jan P.},
 title = {Joint Multi-Scale Tone Mapping and Denoising for HDR Image Enhancement},
 booktitle = {2022 IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (WACVW)},
 year = {2022},
 volume = {},
 number = {},
 pages = {729-738},
 doi={10.1109/WACVW54805.2022.00080}}
}
```

(Full paper: t.ly/TvIU or https://ieeexplore.ieee.org/document/9707563)

## Contact
If you have any questions, please contact Litao Hu at hu430@purdue.edu.
