# Thesis Evaluation

Code used to evaluate the refinement methods investigated in our thesis work: Normalized Convolution Network and Dataset Generation for Refining Stereo Disparity Maps.

Link:  http://liu.diva-portal.org/smash/get/diva2:1333176/FULLTEXT01.pdf

## Info

 3 methods for refining disparity maps were investigated:

1. nconv: Eldesokey et al. (2018) [https://github.com/abdo-eldesokey/nconv](https://github.com/abdo-eldesokey/nconv)
2. sdr: Yan et al. (2019) [https://github.com/danielcranston/SDR](https://github.com/danielcranston/SDR)
3. inpaint: Telea (2004) [https://docs.opencv.org/3.3.1/df/d3d/tutorial_py_inpainting.html](https://docs.opencv.org/3.3.1/df/d3d/tutorial_py_inpainting.html)

These methods were evaluated on 2 datasets:

* middv3: Middlebury V3 training images: Scharstein et al. (2014) [http://vision.middlebury.edu/stereo/data/](http://vision.middlebury.edu/stereo/data/)
* liu: Our own dataset created as part of this thesis.


## Excecuting the code:

1. Download the datasets by following the instructions found in the ```data/``` folders README.md
1. To start the evaluation of a certain method on a dataset, execute ```python evaluate.py -mode MODE -set SET```
    * replace ```MODE``` with 'ncconv', 'sdr' or 'inpaint'
    * replace ```SET``` with 'middv3' or 'liu'


Optional parameters to ```evaluate.py```:

```
-epethresh : sets the threshold of the end-point-error map (in pixels)
             displayed when plotting is turned on. Defaults to 20.
-plot      : sets the plotting mode.  
				0=no plotting
				1=plots appear for each evaluation item
				2=plots appear for each evaluation item, and the figures are saved to disc
```


## Notes

* Evaluation data from NConv is unfortunately only available for Middlebury V3.
* This code was made for my own convenience and does not cater to specific screen resolutions and matplotlib backends
