杨晨的一周报告
--------
<<<<<<< HEAD
1. 完成了webvision的fine-tuning,但是感觉收敛速度太慢了，准备再改动一下
2. 正在阅读：Distilling the Knowledge in a Neural Network。主要讲的是如何从一个冗杂的大模型中，去提炼一个精炼的小模型。用到了soft target和transfer learning。因为第二个例子是用到RNN的语音识别，所以准备学习下如何用RNN去处理语音识别。
3. 因为本周大部分时间都在赶毕设论文，所以进度有点慢。
=======
1. 完成了webvision的baseline，但是后台运行时evulate方法总有BUG，准备放到前台试试
2. 完成阅读：A Survey of Model Compression and Acceleration for Deep Neural Networks
3. 从网上学习了model compression的一些知识包括
> * 基于核稀疏化的方法：都是在训练过程中，对参数的更新进行限制，使其趋向于稀疏，或者在训练的过程中将不重要的连接截断掉
> * 基于模型裁剪的方法：挑选出模型中不重要的参数，将其剔除。而如何找到一个有效的对参数重要性的评价手段，是研究方向点
> * 基于教师——学生网络的方法：教师网络给学生网络指引以训练
> * 基于精细模型设计的方法：使用小卷积核组合来代替大的卷积核来达到压缩和加速的效果
> * 并了解了四种方法最近相关论文的知识概要
>>>>>>> a5b41a000d12e421bab7d063acfb5d06e139a1c2
