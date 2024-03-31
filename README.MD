本代码是为了实现img2txt的自动化过程，这里生成的数据集是yolo格式。首先需要下载对应的依赖。
### 1 Install 
CLIP https://github.com/openai/CLIP 

AlPHA-CLIP https://github.com/SunzeY/AlphaCLIP 


### 2.1 图像在一个文件夹,生成label
    1)运行"../auto_py/dataset.py"划分数据集；
        根据自己的需求修改line 42-46的路径和比例。

    2)运行"../auto_py/auto_label_multi_folder.py",可以生成全种类的txt。
        根据自己的需求修改line 72-103 + 145的权重文件、输入输出路径等。
        注：texts是给alpha_clip的描述词；
            label_dict和text是可视化用的，类别顺序要对应。

    3)运行"../auto_py/bstract.py",提取自己想要的类别。
        根据自己的需求修改line12 + 24-25的classes和输入输出路径。

### 2.2 只针对一个img文件夹想生成txt
    运行"../auto_py/auto_label_one_folder.py"
        根据需求修改line 71-81 + 126-143的路径和自定义类别。


### 3 其他说明
    3.1 已经生成了黑白掩码 想要txt:
        运行"../auto_py/mask2txt/mask2txt.py".

    3.2 针对具体的类别，想用二次验证进一步提升正确率，这里具体问题要具体分析:
        运行“../auto_py/mask2txt/leaf_or_strawberry.py”.

    3.3 想计算生成txt的iou：
        运行"../auto_py/label_val/iou.py". 修改line 45-47的路径 + line 66的类别。

    3.4 想可视化txt里面的坐标：
        运行"../auto_py/label_val/visualization.py". 修改line 5-11的路径即可。

### 之后按照自己想训练的分割网络调整数据集格式即可。

