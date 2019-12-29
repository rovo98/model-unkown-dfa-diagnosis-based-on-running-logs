
## Fault Diagnosis in unknown model DES

使用深度学习方法，基于 DES (离散事件系统) 生成的日志 (running logs) 来进行错误诊断，本质上是一个序列 (sequence) 的分类 (classification) 问题 .

> 即找寻适合该反复分类问题的机器学习方法，完成该任务。重点看如何借鉴已有对文本序列进行分类的例子，CNN (Convolution Neural Network) 或 RNN (Recurrent Neural Network) 可能可行。


### 1. Main Idea

Using one dimensional convolutional neural networks (CNNs), recurrent neural networks (RNNs) or long short term memory (LSTM)

RNNs and LSTM may does better than CNNs for this classification task.

### 2. 1D-CovNets

先尝试使用 1-DCovNets

参考: [https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/](https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/)

当前尝试方案:

刚开始尝试使用产生的数据量较大，编译的模型非常复杂，训练时间非常长（先放弃尝试了）

> 折中方案：选择随机状态大小为 50 ～ 100， 随机生成的日志长度限制为 30 ～ 50， 则预处理编码后产生的矩阵表示为 [1, 50 x (observable_event_set_size + 1)\]
> 产生的日志，经处理后，只剩 3 万多条，用于训练模型。训练时间大概为几分钟。

![1dcovnet_training_example_01](./images/1dcovnet_training_test_01.png)
![1dcovnet_training_example_02](./images/1dcovnet_training_test_02.png)

设置 ``epochs`` 为 100， 跑出来结果 (耗费时间 1 个多小时)

![1dcovnet_training_example_03](./images/1dcovnet_training_test_03_100-epochs.png)

#### CNN tuning

调整 CNN 中调整的超参数 (hyper-parameters)。

##### 1. 一次随意尝试

将 ``kernel_size`` 从 ``3`` 调整为 ``7``， ``pool_size`` 从 ``2`` 调整为 ``5``， 并在拟合模型时将输入的训练集划分一部分 (0.2) 作为 validation set (验证集)。

![1dcovnet_training_example_04](./images/1dcovnet_training_test_03_100-epochs_kernel_pool_size_changed_test_01_model.png)
![1dcovnet_training_example_05](./images/1dcovnet_training_test_03_100-epochs_kernel_pool_size_changed_test_01_training_result.png)

Using gpu to reduce time for training.

![1dcovnet_training_example_06](./images/1dcovnet_training_test_03_100-epochs_kernel_pool_size_changed_test_01_training_result_using_gpu.png)

加载已保存的模型，并从原始日志集中选取若干日志进行测试，来看模型是否能够预测出正确的错误类型:

选取的测试日志 (需要经过压缩编码后，再输入给模型):
![](./images/1dcovnet_training_test_03_100-epochs_prediction_test_01_test_logs.png)

预测情况如下：
![](./images/1dcovnet_training_test_03_100-epochs_prediction_test_01.png)

> NOTICE: 可以看到，所有日志的预测结果都正确，但之前对模型的训练时的评估来看，我们选取的测试日志，很可能就是模型训练集中的样本。
> 因为在数据预处理时，由于每种类别的日志数量不均衡，进行了 over-sampling 以及 under-sampling， 之后再进行打乱 shuffle, 所以从原始的日志上看，无法知道那些是没用于训练的日志。（除非一开始就将原始数据进行划分）
>
> 从模型评估时模型在测试数据集上的表现 (准确率 77% 左右)来看，基本确定我们选取用于测试的数据应该就是包含于它的训练数据集中。🤔

##### 2. tuning number of filters

尝试调整 ``filters`` 数量 (产生的 feature map 数量)，将网络中所有 hyperparameters 调整为原始默认参数，并将 ``epochs`` 下调至 ``10``，减小迭代次数（减少训练所需的时间）
> 为了探索合适的 ``filters`` 大小，我们可以选取一个范围，小于初始值 ``64``的和大于初始值的。

![](./images/1dcovnet_training_test_04_filters_tuning_01.png)
> 完成测试，耗费大概 3 个多小时。 图中，accuracy 准确率是 mean 均值，后面跟着的是 std (standard deviation) 标准差。

![](./images/1dcovnet_training_test_04_filters_tuning_exp_cnn_filters.png)
> 从上图可以看到，随着 ``filters`` feature map 的数量的增加，测试准确率中值（黄色的线）在不断上升，而在 ``64`` 之后反而开始下降，因此，或许 ``64`` 就是合适的值，它同时具备性能和稳定性。
>
> 这么看，模型刚开始选择的 ``64`` 就是比较合适的值。。。

##### 3. tuning kernel size

调整 ``kernel`` (卷积的核或 filter 过滤器大小)，核的大小控制每次读取序列时要考虑的时间步长 (time steps), 然后将时间步长投影 (project) 到 feature map (特征映射，此过程为卷积)。较大的核意味着对输入读取不那么严格。
> 同样，我们可以选择一个范围的 ``kernel_size`` 来进行测试，其中包含初始建立网络选择的值 ``3``。

![](./images/1dconvnet_training_test_05_kernel_size_01.png)
> 完成测试，所花费的时间还是 3 个多小时，测试集准确率是 mean 均值，后面是标准差。

![](images/1dconvnet_training_test_05_kernel_size_01_exp_cnn_kernel.png)
> 从该盒形图中可以明显看出，随着 ``kernel_size`` 的增加，测试准确率中值（黄色线）不断上升，且所有超参数取值对应的测试准确率稳定性非常好。
>
> 从测试来看，``kernel_size`` 取 ``11`` 具有非常不错的效果。
>
> NOTICE: 从图上看，似乎我们还可以尝试取一个比 ``11`` 还大的范围来进行测试，看能否获得更好的效果。
>
> REMARKS: 原因分析：我们知道 kernel size 卷积核的大小是确定时间步长的大小，影响的是对输入序列消息的读取，从效果上看，随着 kernel size 的增大，模型效果越好。可能原因是，输入的数据从原始数据经过编码表示后，是一个维度较高且非常细稀疏的 tensor。适当增大 kernel size 反而能够更好处理这样的数据。

使用比上面 ``11`` 更大的一个范围再次进行测试，结果如下:

![](./images/1dconvnet_training_test_05_kernel_size_02.png)
> 耗时 3 个多小时，可以看到，测试精准率均值都非常高，且它们的稳定性都很好。

![](./images/1dconvnet_training_test_05_kernel_size_02_exp_cnn_kernel.png)
> 从该图中看的话，``kernel`` 取 ``19`` 是最好的。
>
> 因为，随着 kernel size 的增加，测试精准率中值（黄色线）不断上升，意味着，可能还有上升空间，因此可以再次设计实验了测试一组更大的 kernel size。

![](./images/1dconvnet_training_test_05_kernel_size_03.png)
> 耗费 5 个多小时，随着 kernel size 的增大，训练时间变长。

![](./images/1dconvnet_training_test_05_kernel_size_03_exp_cnn_kernel_03.png)
> 从图中，可以看到，随着 kernel size 的增大，测试集精准率中值（黄色线）不断上升，虽然图中有些许离群点 (outlines)。
>
> 表现最好的是 ``27`` 大小 kernel size 的情况。
>
> 从上图看，我们还可以再次设计一个更大的范围进行实验。。。


![](./images/1dconvnet_training_test_05_kernel_size_04.png)
> 耗时将近 4 个小时。

![](./images/1dconvnet_training_test_05_kernel_size_04_exp_cnn_kernel_04.png)
> 相比于之前的值 ``27``, 除了 ``29`` 有两个表现不是很好的离群点外，其他的所有测试结果都比 ``27`` 的要好，这次测试中表现最好的是 ``37``。
>
> 这么看，我们还是可以再设计实验来探索更好的 kernel size 取值。



**需要注意的是，上面的所有调优都是固定其他 hyper-parameter, 然后再探索某个参数的最优值，最后组合形成的效果可能并不是真正的最优。**

**例如：多个参数一起可以形成不同的组合，(如在确定了最优 kernel size 后，再次看在不同 filters 下此 kernel size 的效果) 等等。且实验中，测试是重复 10 次来看稳定性，我们还可以适当提高重复次数，再看看稳定性。**

#### Multi-Channel (head) CNN

multi-head cnn （选择不同 kernel size 的 conv 层做 feature map 的提取，在 flatten 层之后再将它们全部 concatenate 拼接在一起）

##### 1. 简单尝试

网络结构如下:

![](./images/1dconvnet_training_multi_head_01_multi_channel_01.png)
> 三个 head 使用 filters 均为 ``64``, 使用的 kernel size 分别为 ``15``, ``17``, ``19``，且只有一层卷积层。pool_size 均为 ``2``， ``dropout`` 都是 ``0.5``。

由于是一次随意的尝试，先不考虑训练出来的模型的稳定性，优先考虑性能，只做一次训练，结果如下:
![](./images/1dconvnet_training_multi_head_01_multi_channel_01_test.png)
> 训练 10 epochs 后，可以看到效果还行，估计还有很大提升空间，可以在对网络的结构进行调整。

##### 2. multi-channel with 2 conv layers

调整上面的网络结构，使用两层卷积层，并调整三个不同 head 的 kernel size 为 ``17``, ``19``, ``21``。

![](./images/1dconvnet_training_multi_head_02_multi_channel.png)

训练结果如下:

![](./images/1dconvnet_training_multi_head_02_multi_channel_test.png)
> 效果其实和只使用一个 head (kernel size 为 17， 19， 或者 21) 的 1dconvnet 的效果表现其实差不多。
>
> REMARKS: 后续测试可以测试稳定性，以及对不同的 head 的 kernel size 再作出调整。

**NOTICE: 网络结构变复杂后，模型拟合达到好的效果，所需的数据量会大幅度增加，因此我们还可以尝试适当增加 epochs 的数量 (以上实验中使用 epoch 数量都控制在 10 左右，主要想快速衡量模型的优秀程度)**

下面给出的是，一些网络结构调整过后的测试一次测试结果（epochs 不定）

![](./images/1dconvnet_training_multi_head_03_multi_channel_02.png)
![](./images/1dconvnet_training_multi_head_03_multi_channel_02_test.png)

![](./images/1dconvnet_training_multi_head_04_multi_channel_03.png)
![](./images/1dconvnet_training_multi_head_04_multi_channel_03_test.png)

![](./images/1dconvnet_training_multi_head_05_multi_channel_04.png)
![](./images/1dconvnet_training_multi_head_05_multi_channel_04_test.png)

![](./images/1dconvnet_training_multi_head_06_multi_channel_05.png)
![](./images/1dconvnet_training_multi_head_06_multi_channel_05_test.png)

选择上面探索出来的比较好的 filters 和 kernel size 的取值，进行组合。

![](./images/1dconvnet_training_multi_head_07_multi_channel_06.png)
![](./images/1dconvnet_training_multi_head_07_multi_channel_06_test.png)
> 效果是好了一点。

### 3. RNNs

#### Simple RNN or GRN

#### LSTM

可先尝试只使用 LSTM 的方案。

Full CNNs with LSTM 

> reference: [https://paperswithcode.com/paper/multivariate-lstm-fcns-for-time-series](https://paperswithcode.com/paper/multivariate-lstm-fcns-for-time-series)

## Issues

1. 如何更好的表示日志观察序列？

当前问题，即如何使用数据，表示数据 (表示学习，Representation learning), 特征工程 (feature engineering)。

可以先尝试自己使用传统的机器学习方法，用 MLP (multi-layer Perceptron network) 前馈网络来进行实验。之后再参考他人应用于文本序列分类的模型（通常是深度学习方法，即让模型自动做表示学习，自动抽取高层特征）。

> 放弃尝试说手动进行特征抽取的方式，例如对于文本分类，Bow (Bag of words) 需具备一定的专业知识专家才能来定所要使用的 vocabulary。

参考 **A Compact Encoding for Efficient Character-level Deep Text Classification-marinho2018** 中 Character-Level (字符级别) 紧凑编码来处理输入的日志中的观察。之后再考虑配合 CNNs 或 RNNs 或者 LSTM (Long Short Term Memory) 模型来训练。

2. 产生的日志类别数量不平衡问题，如何解决？

对数量较多的类别使用 under sampling (欠采样)，以减少该类别训练数据的数量，而对数量少的类别样本使用 over sampling (过采样)，适当重复一些样本，以增加该类别的样本数量。