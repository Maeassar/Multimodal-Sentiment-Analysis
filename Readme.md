## AI Lab5: 多模态情感分析（Multimodal Sentiment Analysis）
<hr/>

### Introduction
<hr/>
本实验考虑文本和图像两个维度的特征，来进行一个双模态情感分析的三分类任务， 分类标签为positive, negative, neutral。

### Structure
<hr/>

```
C:.
│  Config.py #默认参数设置
│  createresult.py #生成结果
│  data_processing.py #数据预处理
│  main.py #项目运行文件
│  Readme.md #项目说明
│  requirements.txt #环境文件
│  test_result.txt #结果
│
│
├─bert_multilingual #"bert-base-multilingual-cased"模型（#改模型比较大 请自行下载）
│      config.json
│      pytorch_model.bin
│      vocab.txt
│
├─dataset #数据集
│  │  test.json
│  │  test_without_label.txt
│  │  train.json
│  │  train.txt
│  │  val.json
│  │
│  └─data #该文件较大，请自行下载
├─Lossdata #记录模型的loss
│      ImprovedMultimodel_loss.csv #代码自动生成
│      SimpleMultimodel_loss.csv #代码自动生成
│
├─Metrics #记录模型的metrics
│      ImprovedMultimodel_metrics.csv #代码自动生成
│      SimpleMultimodel_metrics.csv #代码自动生成
│
├─models #模型
│  │  ImprovedMultimodel.py
│  │  SimpleMultimodel.py
│  │
│  └─__pycache__
│          ImprovedMultimodel.cpython-39.pyc
│          SimpleMultimodel.cpython-39.pyc
│
├─Savemodel #保存的模型
│      ImprovedMultimodel.pth #代码自动生成
│      SimpleMultimodel.pth #代码自动生成
│
└─__pycache__
        Config.cpython-39.pyc

```

### Setup
<hr/>
本项目所使用的环境为python3.9(pytorch2023),

- chardet==5.2.0
- pandas==2.1.4
- transformers==4.32.1
- torch==1.13.1
- torchvision==0.14.1
- scikit-learn==1.3.0


##### 您可以直接运行下列代码安装requirements.txt, 实现环境的配置


```
pip install --upgrade pip 
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt
```
### Run pipeline for train
<hr/>
本实验使用argparse包进行模型和超参数的选择，可选的参数如下：

- --do_train 选择后训练模型
- --fuse_model_type 选择模型
- --epoch 调整训练轮次
- --do_val_test 选择后对验证集进行测试
- --do_test 选择后对进行测试
- --text_only 消融实验-只文本
- --img_only 消融实验-只图像

```python
python main.py --do_train --fuse_model_type ImprovedMultimodel --epoch 5
```

### Run pipeline for test

如果您想在验证集上对模型进行测试：

```python
python main.py --do_val_test --fuse_model_type ImprovedMultimodel
```

如果您想进行消融实验：：

```python
python main.py --do_val_test --fuse_model_type ImprovedMultimodel --img_only(or --text_only)
```

如果您想在测试集上预测标签：

```python
python main.py --do_test --fuse_model_type ImprovedMultimodel
```
### Reference
<sup><a href="#ref1">1</a></sup>  <span name = "ref4">[DaSE_ContemporaryAI_22Spring/5_Multimodal_Sentiment_Analysis_Model at main · younghojan/DaSE_ContemporaryAI_22Spring (github.com)](https://github.com/younghojan/DaSE_ContemporaryAI_22Spring/tree/main/5_Multimodal_Sentiment_Analysis_Model)</span> 

<sup><a href="#ref1">2</a></sup>  <span name = "ref4">[attention-is-all-you-need-pytorch/transformer/Models.py at master · jadore801120/attention-is-all-you-need-pytorch (github.com)](https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py)</span>