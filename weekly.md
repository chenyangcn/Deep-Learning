杨晨的一周报告
--------
1. 又看了一遍Distilling the knowledge的论文，然后在网上也查了些资料，发现这种方法对于label偏少或者label之间的关联性很小的时候表现并不能有很大的提高。现在猜想是不是cifar10中10个类别太少且关联性不大的问题。
2. 正在用cifar100进行再一步实验，代码已经写完，正在进行参数的调整。
3. 完善了了inceptionv2的webvision代码，且用inceptionv2在imagenet测试集上目前跑出了72.120%的准确率
![inceptionv2 loss](./PyTorch/imagenet/inceptionv2_loss.png)
![inceptionv2 acc](./PyTorch/imagenet/inceptionv2_acc.png)
4. 正在看PAYING MORE ATTENTION TO ATTENTION:IMPROVING THE PERFORMANCE OF CONVOLUTIONAL NEURAL NETWORKS VIA ATTENTION TRANSFER（ICLR2017）这篇论文讲的是：通过恰当地定义卷积神经网络的注意力，使用这种类型的信息来显著提高学生CNN网络的性能，迫使学生网络模仿一个强大的教师网络的注意力map。
