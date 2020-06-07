### bugs
1. Resize, rather than Crop


### Question
1. How to compute Loss?
2. 用不用ignore?

### 经验
1. 如果模型正确，那么开始几个epoch loss下降会非常快
2. Data预处理时，先输出一下图片的维度，并打印一下图片看看
3. 0 is the label of background, 255 is the label of boarder
