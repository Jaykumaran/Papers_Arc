## Try 1:
Configs: `lr = 0.001`

### TEST DATASET:

| Batch Size | Accumulation Steps | mAP    | Mem                   |
|------------|--------------------|--------|-----------------------|
| 8          | 1                  | 0.7189 | ~3GB                  |
| 32         | 8                  | 0.40   | ~9600MiB / 12288MiB   |
| 32         | 1                  |        | ~9022MiB / 12288MiB   |

## Try 2:
Configs: `lr = 0.01`

<img src = "https://github.com/Jaykumaran/Papers_Arc/blob/main/ssd_scratch/training_loss.png">









# Single Shot Detector (SSD)
=======



References: 
1. https://youtu.be/c_nEue9itwg?feature=shared
2. https://github.com/explainingai-code/SSD-PyTorch/


# SSD - Default Boxes or Anchor boxes

There will be k ref boxes per grid.

The convolution takes care of :
For eg:
38x38xK1 - classification scores
along with transformation prediction `[t_{cx}, t_{cy}, t_w, t_h]`

- $t_{cx} = (G_{cx} - D_{cx}) / D_w$
- $t_{cy} = (G_{cy} - D_{cy}) / D_h$
- $t_w = \log(G_w / D_w)$
- $t_h = \log(G_h / D_h)$

---

$s_k = s_{min} + ((s_{max}-s_{min}) / (m-1)) (k-1)$,  k belongs [1, m]

$s_{min} = 0.2 ; s_{max} = 0.9$

The largest feature map will have $s_{min}$
The smallest feature map will have $s_{max}$

Inbetween feature maps as equally spaced scales.

Aspect ratios $a_r = \{1, 2, 1/2, 3, 1/3\}$

Therefore,
* $w = s_k * \sqrt{a_r}$
* $h = w_k / \sqrt{a_r}$

For a_r = 1  

$a_r = 1 \& \sqrt{s_k s_{k+1}}$  where $s_{k+1}$ is the scale of next feature map. For the last feature map i.e. s = 0.9, k = 5 here, we will use $s_{k+1}$ as 1:

Additional default boxes:

Especially for the first aspect ratio,

|                 | k=1                 | k=2                 | k=3                 | k=4                 | k=5                 |
|-----------------|---------------------|---------------------|---------------------|---------------------|---------------------|
| scale           | 0.2                 | 0.375               | 0.55                | 0.725               | 0.9                 |
| $a_r = 1$       | w=0.2, h=0.2        | w=0.375, h=0.375    | w=0.55, h=0.55      | w=0.725, h=0.725    | w=0.9, h=0.9        |
| $a_r = 2$       | w=0.28, h=0.14      | w=0.53, h=0.265     | w=0.78, h=0.39      | w=1.02, h=0.51      | w=1.27, h=0.63      |
| $a_r = 0.5$     | w=0.14, h=0.28      | w=0.265, h=0.53     | w=0.39, h=0.78      | w=0.51, h=1.02      | w=0.63, h=1.27      |
| $a_r = 1 \& \sqrt{s_k s_{k+1}}$ | w=0.27, h=0.27 | w=0.45, h=0.45 | w=0.63, h=0.63 | w=0.81, h=0.81 | w=0.95, h=0.95 |



The centre locations of each of the grid cell will be For eg: take a 3x3 feature map

here, $f_k$ = 3

$\frac{i+0.5}{|f_k|}, \frac{j+0.5}{|f_k|}$

$i, j \in [0, |f_k|)$, i.e. i, j can take values between [0, 1, 2]



| Feature Map Size | 38 x 38 | 19 x 19 | 10 x 10 | 5 x 5 | 3 x 3 | 1 x 1 |
|---|---|---|---|---|---|---|
| # Boxes * Aspect Ratios | 1444 * k1 | 361 * k2 | 100 * k3 | 25 * k4 | 9 * k5 | k6 |
| Output 1 | c classification scores | c classification scores | c classification scores | c classification scores | c classification scores | c classification scores |
| Output 2 | 4 box parameters | 4 box parameters | 4 box parameters | 4 box parameters | 4 box parameters | 4 box parameters |


### SSD Matching Strategy

Foreground:

- Best default boxes is filtered to match the GT boxes.
- Other default boxes with overlap of > 0.5 with any GT box. Also labelled as FG

Background: 
- Other than these all boxes are BG default boxes.


But BG is hell more than FG, the network will have hard time learning.

 so here comes to save, 

 "Hard Mining" :  Top K BG default boxes with higest confidence loss is chosen , around K = 3*number of FG default boxes. Choose these only for training.


### SSD training Loss:


> $Loss = \frac{1}{N} (loss_{cls} + loss_{loc})$

where, 

N - number of foreground matched default boxes

Positive FG boxes, and selected (hard mining) BG default boxes

> $loss_{cls} = loss_{cls}(pos) + loss_{cls}(neg)$

Sum of losses over all FG default boxes, compute loss for four transformation matrixes and sum them.
> $loss_{loc} = \sum_{i \in Pos}^{N} \sum_{m \in \{cx,cy,w,h\}};  \text{smooth}_{L1}(l_i^m - \hat{g}_j^m)$

- $\hat{g}_j^{cx} = (g_j^{cx} - d_i^{cx}) / d_i^w$

- $\hat{g}_j^{cy} = (g_j^{cy} - d_i^{cy}) / d_i^h$

- $\hat{g}_j^{h} = \log \left( \frac{g_j^{h}}{d_i^{h}} \right)$

- $\hat{g}_j^{w} = \log \left( \frac{g_j^{w}}{d_i^{w}} \right)$




### MODEL ARCHITECTURE

For fine-tuning for detection get rid of classification layer,

Take a VGG16 , input size is 300x300

where `conv4_3` will have `38x38` and `conv5_3` will output `19x19` feature maps.

The fc layers are convered from linear layers, where fc6 4096x (512 x 7 x7) to 1024 x 512 x 3 x 3. Similarly fc7 4096 x 4096 to 1024 x 1024 x 1 x 1.

For fc6, additional dilation is used. n




For  a_r = 1, 2, 0.5 for first, last and last before  


---

Total : **8732 default boxes**






```
/home/jaykumaran/Downloads/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007
├── Annotations
├── ImageSets
│   ├── Layout
│   ├── Main
│   └── Segmentation
├── JPEGImages
├── SegmentationClass
└── SegmentationObject

8 directories
```
