# ESPNetv2-paddle

采用paddle和paddleseg实现ESPNetv2分割模型的复现

## 环境

python==3.7

请不要使用3.8，可能会导致以下一段代码出错，具体原因可以参考paddle的一个未解决的[issue](https://github.com/PaddlePaddle/Paddle/issues/26174)

    copy.deepcopy(Conv(features, features, 3, 1, groups=features))

paddlepaddle >= 2.1

其余与paddleseg一致

## 训练过程分享

参数文件都在benchmark/espnetv2_config目录下，请按如下顺序使用

espnetv2_iter80k.yml $\Rightarrow$ espnetv2_lovasz_40k.yml $\Rightarrow$ espnetv2_finetune_40k.yml

首先，用espnetv2_iter80k.yml进行第一步训练。我们将作者提供的在imagenet-1k数据集上训练出的分类模型从torch转至paddle，作为预训练的backbone。根据论文给出的方法，我们先采用分辨率较小的图片（在cityscapes上即为512 * 256）进行初步的训练。

注：在复现的过程中我们发现paddle和torch的conv2d在stride=2，且padding>0时边缘的结果存在差异，所以作者提供的模型可能并不能帮助我们达到最完美的效果，或许用paddle从imagenet从头训练一个backbone会带来更好的效果。

我们在训练的过程中可以观察到部分类别在早期的miou为0，随着学习率的减小这些类别的指标才开始上升，推测主要原因是数据集的类别不均衡导致的，所以我们在espnetv2_lovasz_40k.yml采用lovasz loss进行训练，对这一问题进行针对性解决。

最后，在espnetv2_finetune_40k.yml中，我们用DiceLoss和BootstrappedCrossEntropyLoss 对模型进行最后的finetune，达到了单尺度miou67.45%，多尺度miou69.06%的结果，单尺度miou比原作者所给出的精度高出1%。

## evaluate

| 模型 | backbone | miou | miou(ms+flip) | backbone链接 | model链接 |
|-------|----------|------|---------------|--------------|-----------|
|ESPNetv2_Seg|ESPNetv2|0.6745|0.6906|[百度网盘(提取码: 63tf)](https://pan.baidu.com/s/1HAqo9JTawwQtVlj2gPKY2g)|[百度网盘(提取码: 63tf)](https://pan.baidu.com/s/1HAqo9JTawwQtVlj2gPKY2g)|