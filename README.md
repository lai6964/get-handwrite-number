# 生成一些简单的算式


1. 下载手写公式数据集 CRHOME2019_data, 解压CROHME2019_data/Task1_onlineRec/subTask_symbols/Train，运行convertInkmlToImg.py将inkml墨迹转为png图片

python convertInkmlToImg.py

2.运行get_simple_symbol.py抽取你想提取字符路径及标签

python get_simple_symbol.py

3.下载MNIST数据集，提取保存为单张图片

python get_mnist.py

4.运行conbine_num_sym.py进行拼接，得到简单的算式

python conbine_num_sym.py

