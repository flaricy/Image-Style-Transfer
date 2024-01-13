# CV-Project----Image-Style-Transfer

## 使用方法
- 请将 main.py 中第8行 `device = torch.device("mps")` 改为合适的gpu。
- 修改 main.py 中 165、166 行以更换content和style。注意相对路径。
- main.py 第161行修改生成的图片尺寸。
- 在 `get_inits` 函数中修改初始化方式。
- 关于误差系数的超参数在第95行。
- 使用了tensorboard来记录训练过程。
- 运行 python3 main.py 即可。
- 运行tensorboard的命令: tensorboard --logdir=runs


### 关于图片
- 欢迎在images和styles中添加更多图片，训练结果在results文件夹下。

### 其他
- vgg_architecture是VGG-19的架构，方便查询。
- dev_log 暂时没写。