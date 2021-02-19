# Grey and white matters segmentation on exvivo DWI mouse data

![alt text](seg_gm_wm.gif)

Segmentation model for mouse spinal gray and white matter on DWI data.

Created with [ivadomed](http://ivadomed.org/) using data from University of Queensland 🇦🇺.
Training features include: 2D unet, multiclass, SoftSeg.

This model is readily available in [SCT](https://spinalcordtoolbox.com/en/stable/) (v5.1.1 and higher)

... can be installed as follows:
```bash
sct_deepseg -install-task seg_mice_gm-wm_dwi
```

... and can be used as follows:
```bash
sct_deepseg -i NIFTI_IMAGE -task seg_mice_gm-wm_dwi
```

Any question? Please feel free to post on [SCT forum](https://forum.spinalcordmri.org/c/sct/8). 
