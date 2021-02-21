# Soft Compression for Lossless Image Coding

## [[Paper]](https://arxiv.org/pdf/2012.06240.pdf) [[Citation]](#citation)

### Abstract

Soft compression is a lossless image compression method, which is committed to eliminating coding redundancy and spatial 
redundancy at the same time by adopting locations and shapes of codebook to encode an image from the perspective of 
information theory and statistical distribution. In this paper, we propose a new concept, compressible indicator 
function with regard to images, which gives a threshold about the average number of bits required to represent a 
location and can be used for revealing the performance of soft compression. We investigate and analyze soft compression 
for binary image, gray image and multi-component image by using specific algorithms and compressible indicator value. 
In terms of compression ratio, soft compression algorithm outperforms the popular classical standards PNG and JPEG2000 
in image lossless compression. It is expected that the bandwidth and storage space  needed when transmitting and storing 
the same kind of images can be greatly reduced by applying soft compression.

### Framework and Role

<div align="center">
  <img src='Figures/Encoder.png' width="100%"/>
</div>

## Prerequisites for Code

Required packages:
```
pip install opencv-python --user
pip install opencv-contrib-python --user
pip install numpy
```

Datasets:

```
Fashion_MNIST: https://github.com/zalandoresearch/fashion-mnist
CIFAR-10: http://www.cs.toronto.edu/~kriz/cifar.html
```


##### Notes
- We tested this code with Python 3.8 and opencv 4.0.1.
- The code also works with other versions of the packages.


## Main Modules
- To train a model yourself, you have to first prepare the data as shown in [Prerequisites for Code](#Prerequisites for Code).
- Then, put them into foler 'Dataset\\Fashion_MNIST' and 'Dataset\\CIFAR10'
- After preparation, you can run 'MainProgram.py' to get the cross results by using soft compression algorithm for gray image
- Run 'MainProgram2.py' to get the single results by using soft compression algorithm for gray image

| Module | Effect of running|
| --- | --- |
| PreProcess       | `Generate the training set and testing set` |
| ShapeFinding     | `Find the frequency set` |
| CodeProcessing   | `Generate the codebook according to the frequency` |
| Encode           | `Encode images by using this algorithm` |
| Decode           | `Decode binary stream data into the original image` |
| MainProgram      | `Get the cross results by using soft compression algorithm for gray image` |
| MainProgram2     | `Get the single results by using soft compression algorithm for gray image` |

## Other Information
**NOTE**: The program may take a long time. We run it with multiprocessing. You can choose optional methods to accelerate it

## Future Work

- Iterative soft compression algorithm makes it have better performance
- Optimize program and improve efficiency

## Citation

If you use the work released here for your research, please cite this paper:
```
@article{xin2020soft,
  title={Soft Compression for Lossless Image Coding},
  author={Xin, Gangtao and Fan, Pingyi},
  journal={arXiv preprint arXiv:2012.06240},
  year={2020}
}
```
