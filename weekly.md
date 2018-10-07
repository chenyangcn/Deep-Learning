杨晨的一周报告
--------
1. 使用bolei zhou的Learning Deep Features for Discriminative Localization论文中的CAM方法去提取正确类别的attention去指导mask生成（还是只用到了label的信息，指导mask训练的attention用的是label和feature map生成的）。现在可以准确找出鸟的区域，且单加attention的准确率为85.9%+（虽然有的分类结果不好，但是鸟的区域还是可以框出来）  
#### 原始图片  
![Original Picture](./original.jpg)  
#### 加了mask的图片  
![Attention Picture](./attention.jpg)
