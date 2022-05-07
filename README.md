# Gesture-Recognition-with-CNNs

Download the data set from the following link:http://web.csie.ndhu.edu.tw/ccchiang/Data/All_gray_1_32_32.rar

Method Description and Comparing with Hyperparameters and Model Architectures
---
***In the intial setting, I constructed the model with two sequences of network, including a convolution layer, a activation function , a pooling layer in each sequence.***

```python==
import torch
import torch.nn as nn

# Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 16, 16)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 16, 16)
            nn.Conv2d(
                in_channels=16,             # input height
                out_channels=32,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,
            ),                              # output shape (32, 16, 16)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # output shape (32, 8, 8)
        )
        self.out = nn.Linear(32 * 8 * 8, 3)   # fully connected layer, output 3 classes
```
---
### Situation 1 : The Difference of Hyperparameters
| Epoch | Optimizer | Accuracy of Training(%) | Accuracy of Testing(%) |
|:----------:|:---------:|:--------------------:|:-------------------:|
|     25     |    SGD    |48.7|54.17|
|     25     |  Adagrad  |84.44|73.89|
|     25     |  RMSProp  |86.85|77.5|
|     25     |   Adam    |96.11|86.94|
|     50     |    SGD    |51.3|44.72|
|     50     |  Adagrad  |79.63|67.78|
|     50     |  RMSProp  |96.48|82.22|
|    **50**     |   **Adam**    |**99.44**|**93.89**|
---
### Situation 2 : The Difference of Model Architectures of Sequence
***Based on the result above the chart, I set the parameter with epoch=50, optimizer=Adam.***

```python==
import torch
import torch.nn as nn

# Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 16, 16)
        )
        self.out = nn.Linear(16 * 16 * 16, 3)   # fully connected layer, output 3 classes
```

| Sequence | Accuracy of Training(%) | Accuracy of Testing(%) |
|:--------:|:-----------------------:|:----------------------:|
|    1     |          96.3           |         68.33          |
|  **2**   |        **99.4**         |       **93.89**        |

***Without a doubt, the accuracy of testing with two sequences of network is better than one's. However, for convenience of experimentation, I utilized one sequence of network for the next comparation.***

---
### Situation 3 : The Difference of Model Architectures of inner parameters

* ***Number of Feature Filter :***

| n_filiter | Accuracy of Training(%) | Accuracy of Testing(%) |
|:--------:|:-----------------------:|:----------------------:|
|    8     |          95.56           |         69.17         |
|  16   |        97.22         |       73.06      |
|  32   |        98.33         |       76.67     |
|  **64**   |        **100**         |       **84.72**      |
|  128   |        100         |       91.39      |

***The more number of filter, the more accuracy of testing will be possibility higher, however, the time waste and burden of hardware also increase, finally, I chose n_filiter=64 for the parameter in neruon.*** 
* ***Filter Size :***

| Kernel Size | Accuracy of Training(%) | Accuracy of Testing(%) |
|:--------:|:-----------------------:|:----------------------:|
|    3     |          96.67           |         65.56         |
|  **5**   |        **98.33**         |       **76.67**      |

* ***Kernel Size of Pooling Layer :***

| Kernel Size | Accuracy of Training(%) | Accuracy of Testing(%) |
|:--------:|:-----------------------:|:----------------------:|
|    2     |          99.81           |         83.89         |
|  **4**   |        **96.67**         |       **87.5**      |

***In the result, the inner parameters, number of feature filter=64, filter size=5, kernel size of pooling layer=4, are adopted in the network.***

***Finally, Number of feature filter and kernel size of pooling layer are assigned to two sequences of network.***

```
Loss at Epoch 0 is 1.2168701399456372
Loss at Epoch 1 is 1.0974623289975254
Loss at Epoch 2 is 1.096157973462885
Loss at Epoch 3 is 1.0905030098828403
Loss at Epoch 4 is 1.077196717262268
Loss at Epoch 5 is 1.0530863295901904
Loss at Epoch 6 is 1.0225716124881397
Loss at Epoch 7 is 0.9748118248852816
Loss at Epoch 8 is 0.9226068691773848
Loss at Epoch 9 is 0.8592216318303888
Loss at Epoch 10 is 0.799804999069734
Loss at Epoch 11 is 0.7548064535314386
Loss at Epoch 12 is 0.7122143073515459
Loss at Epoch 13 is 0.680005363442681
Loss at Epoch 14 is 0.6488951552997936
Loss at Epoch 15 is 0.6219885457645763
Loss at Epoch 16 is 0.5968957136977803
Loss at Epoch 17 is 0.5709817626259543
Loss at Epoch 18 is 0.5488803711804476
Loss at Epoch 19 is 0.5227069610899145
Loss at Epoch 20 is 0.5009708120064302
Loss at Epoch 21 is 0.4788070619106293
Loss at Epoch 22 is 0.4572262628511949
Loss at Epoch 23 is 0.4384942759167064
Loss at Epoch 24 is 0.4169470979408784
Loss at Epoch 25 is 0.4000031148845499
Loss at Epoch 26 is 0.37714689428156073
Loss at Epoch 27 is 0.35991076650944626
Loss at Epoch 28 is 0.3390561254187064
Loss at Epoch 29 is 0.320497564971447
Loss at Epoch 30 is 0.3013303909789432
Loss at Epoch 31 is 0.28368994254957547
Loss at Epoch 32 is 0.2666636620732871
Loss at Epoch 33 is 0.24946328218687663
Loss at Epoch 34 is 0.23689039288596672
Loss at Epoch 35 is 0.21950358321720903
Loss at Epoch 36 is 0.20938315479592842
Loss at Epoch 37 is 0.19384147023612802
Loss at Epoch 38 is 0.18616960079155184
Loss at Epoch 39 is 0.17104969986460425
Loss at Epoch 40 is 0.16506836072287775
Loss at Epoch 41 is 0.1514935181899504
Loss at Epoch 42 is 0.14437737468291412
Loss at Epoch 43 is 0.13537038016048344
Loss at Epoch 44 is 0.12659668262031945
Loss at Epoch 45 is 0.1214209772138433
Loss at Epoch 46 is 0.11186013988811862
Loss at Epoch 47 is 0.10835131435570391
Loss at Epoch 48 is 0.09982061428441243
Loss at Epoch 49 is 0.09737169937315313
=================================================
Epoch is 50
-------------------------------------------------
The batch size is 50
-------------------------------------------------
The number of feature filter is 32 , size is 5
-------------------------------------------------
The Pooling Size is 2
-------------------------------------------------
The optimizer is Adam
-------------------------------------------------
The loss function is CrossEntropyLoss
-------------------------------------------------
Check Accuracy of Training : 
Got 540 / 540 with accuracy 540.0/540.0 = 100.0
-------------------------------------------------
Check Accuracy of Testing : 
Got 343 / 360 with accuracy 343.0/360.0 = 95.28
```

![](https://i.imgur.com/PVEaYi2.png)
