
### bugs
1. Resize, rather than Crop


### Question
1. How to compute Loss? 把三维图像展开成2维，用crossentropy
2. 用不用ignore? 不用
3. 为什么用了DataParallel后边的很慢？none
4. metric怎么计算的？
5. 找更牛逼的模型，原来模型理论效果很好？
6. 更好的criterion？
7. 写log写到了一个文件上，无法同时运行多个model的程序? done

### 经验
1. 如果模型正确，那么开始几个epoch loss下降会非常快
2. Data预处理时，先输出一下图片的维度，并打印一下图片看看
3. 0 is the label of background, 255 is the label of boarder
