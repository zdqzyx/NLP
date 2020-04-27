# TextClassification

主要有：
- FastText
- TextCNN
- TextBiRNN
- TextBiRNNAtt
- TextSelfAtt
- TextTransformerEncoder

## [TextRNN](./imp_by_tensorflow2/TextCNN/textcnn.py)

> TextCNN原始论文： [Convolutional Neural Networks for Sentence Classification](http://www.aclweb.org/anthology/D14-1181) 

### TextCNN 的网络结构：

![](https://raw.githubusercontent.com/zdqzyx/images/master/blog/text_classification_images/TextCNN_network_structure.png)


## [BiRNN+Attention](./imp_by_tensorflow2/TextBiRNNAtt/text_birnn_att.py)

### 论文
> 此处对于注意力机制的实现参照了论文 [Feed-Forward Networks with Attention Can Solve Some Long-Term Memory Problems](https://arxiv.org/pdf/1512.08756.pdf)

### 此处实现的网络结构：

![](https://raw.githubusercontent.com/zdqzyx/images/master/blog/text_classification_images/TextAttBiRNN_network_structure.png)

> 这里实现的Attention是将GRU各个step的output作为key和value，增加一个参数向量W作为query，主要是为了计算GRU各个step的output的权重，最后加权求和得到Attention的输出


## [TranformerEncoder](./imp_by_tensorflow2/TextTransformerEncoder/modeling.py)

### 参考论文
> [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

### 参考实现
> [tensorflow2.0 offical tutorials/text/transformer](https://www.tensorflow.org/tutorials/text/transformer)

### 此处实现的网络结构：
![](https://raw.githubusercontent.com/zdqzyx/images/master/picgo/20200427102052.png)


