杨晨的一周报告
--------
1. 发现网络的问题在于提特征的C3D网络效果就不是很好，单独用C3D去测试效果就很不好，然后重新训练了一个，没有达到他们论文中的效果，差10个点。然后用自己的方法在此基础上训练，效果仅仅提升了一点点1-2个点（相较于自己训练出来的网络）

2. 看了下PANnets这篇论文，PANnets相比于MASK-RCNN做了3点改进。分别是：1.在FPN上缩短了从level feature map上缩短了到high level feature map的距离，可以让空间特征更好的指导网络选取anchor。2.在ROI pooling阶段加入了一个自适应的fetature map选取操作，让网络自己选择使用哪一个feature map上的特征。3.修改了产生instance segmentation部分的网络。
