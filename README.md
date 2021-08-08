# ESPNetv2-paddle

ESPNetv2 implementation using paddlepaddle and paddleseg

## environment

python == 3.7

Do not use python3.8. We find the following code not supported when using python3.8. For more details, please read the following [issue](https://github.com/PaddlePaddle/Paddle/issues/26174)

    copy.deepcopy(Conv(features, features, 3, 1, groups=features))

paddlepaddle >= 2.1

others are the same as paddleseg

## training tricks

The configs are put in benchmark/espnetv2_config. please use it in the following order.

espnetv2_iter80k.yml $\Rightarrow$ espnetv2_lovasz_40k.yml $\Rightarrow$ espnetv2_finetune_40k.yml

First, using espnetv2_iter80k.yml, we use a smaller image resolution for training (512 × 256
for the CityScapes dataset). we use the backbone pretrained on imagenet-1k provided by the author of ESPNetv2.(As the conv2d function is different when stride > 1 and padding > 0 in paddle and torch, the pretrained model provided by the author, which is trained using torch, is not perfect for our implementation. Maybe using paddle to train a new backbone can further improve the performance)

Then, using espnetv2_lovasz_40k.yml, we use the lovasz loss to solve the problem of the unbalanced labels.

Finally, using espnetv2_finetune_40k.yml, we use DiceLoss and BootstrappedCrossEntropyLoss to finetune.

It cost about 30 hours to train the model using v100 provided by AIstudio

## evaluate

| model | backbone | miou | miou(ms+flip) | backbone_url | model_url |
|-------|----------|------|---------------|--------------|-----------|
|ESPNetv2_Seg|ESPNetv2|0.6745|0.6906|[百度网盘(提取码: 63tf)](https://pan.baidu.com/s/1HAqo9JTawwQtVlj2gPKY2g)|[百度网盘(提取码: 63tf)](https://pan.baidu.com/s/1HAqo9JTawwQtVlj2gPKY2g)|

