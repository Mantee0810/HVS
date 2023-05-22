# HVS
这是一个**即插即用**的模块，自适应不同大小的输入（甚至连**通道**都是自适应的）
## 用法
在模型的类定义中，按以下方式处理：
* from HVSModule.py import HVSModule
* 在__init__函数中，加入`self.hvs = HVSModule()`
* 在forward函数中，加入`map = self.hvs(map)`
**就大功告成啦**

下面是fine-tune时间，请自由发挥！
