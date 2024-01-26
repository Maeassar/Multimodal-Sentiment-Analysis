from torchvision.transforms import transforms
from tqdm import tqdm
from PIL import Image
import json
import os
import torch
import argparse
from Config import config
from torch.utils.data import Dataset, DataLoader, RandomSampler
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--do_train', action='store_true', help='training')
parser.add_argument('--fuse_model_type', default='SimpleMultimodel', help='融合模型类别', type=str)
parser.add_argument('--lr', default=5e-5, help='设置学习率', type=float)
parser.add_argument('--weight_decay', default=1e-2, help='设置权重衰减', type=float)
parser.add_argument('--epoch', default=4, help='设置训练轮数', type=int)

parser.add_argument('--do_val_test', action='store_true', help='val-test')
parser.add_argument('--text_only', action='store_true', help='消融实验：文本预测')
parser.add_argument('--img_only', action='store_true', help='消融实验：图像预测')

parser.add_argument('--do_test', action='store_true', help='test')
parser.add_argument('--load_model_path', default=None, help='已经训练好的模型路径', type=str)

args = parser.parse_args()
config.learning_rate = args.lr
config.weight_decay = args.weight_decay
config.epoch = args.epoch
config.fuse_model_type = args.fuse_model_type
config.load_model_path = args.load_model_path
print(args.img_only)
if args.img_only:
    print("u know img_only")
    config.only = 'img'
elif args.text_only:
    config.only = 'text'
print("config", config.only)
print('TextModel: {}, ImageModel: {}, FuseModel: {}'.format('bert-multilingual-cased', 'ResNet50', config.fuse_model_type))
device = torch.device('cpu')


def get_image_transforms():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.46777044, 0.44531429, 0.40661017],
                std=[0.12221994, 0.12145835, 0.14380469],
            ),
        ]
    )

def read_from_file(path, data_dir, only=None):
    data = []
    print("into data only", only)
    with open(path) as f:
        json_file = json.load(f)
        for d in tqdm(json_file, desc='Reading from json----- [Loading]'):
            guid, label, text = d['guid'], d['label'], d['text']
            if guid == 'guid': continue

            if only == 'text':
                img = Image.new(mode='RGB', size=(224, 224), color=(0, 0, 0))

            else:
                img_path = os.path.join(data_dir, (guid + '.jpg'))
                img = Image.open(img_path)
                img.load()

            if only == 'img':
                text = ''
            data.append((guid, text, img, label))
        f.close()

    return data

def save_model(output_path, model_type, model):
    output_model_dir = os.path.join(output_path, model_type)
    if not os.path.exists(output_model_dir): os.makedirs(output_model_dir)    # 没有文件夹则创建
    model_to_save = model.module if hasattr(model, 'module') else model     # Only save the model it-self
    output_model_file = os.path.join(output_model_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)

current_dir = os.getcwd()
train_data_processed = read_from_file(current_dir + "/dataset/train.json", current_dir + "/dataset/data", config.only)
val_data_processed = read_from_file(current_dir + "/dataset/val.json", current_dir + "/dataset/data", config.only)
test_data_processed = read_from_file(current_dir + "/dataset/test.json", current_dir + "/dataset/data", config.only)
label_mapping = {"positive": 0, "negative": 1, "neutral": 2}#label映射

class CustomDataset(Dataset):
    def __init__(self, data, image_transform=None):
        self.data = data
        self.image_transform = image_transform

        self.tokenizer = BertTokenizer.from_pretrained("./bert_multilingual")
        #self.bert_model = BertModel.from_pretrained("./bert_multilingual")

        self.max_text_length = 64

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        guid = sample[0]
        text = sample[1]
        image = sample[2]
        label = sample[3]
        if pd.isna(label):
            label_mapped = 3
        else:
            label_mapped = label_mapping[label]
        # 处理image
        if self.image_transform:
            image = self.image_transform(image)
        # 处理text
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=64, return_token_type_ids=True)

        text_ids = inputs['input_ids'].squeeze(0)
        #print("text_ids", text_ids)
        attention_mask = inputs['attention_mask'].squeeze(0)
        token_type_ids = inputs['token_type_ids'].squeeze(0)

        text_ids = F.pad(text_ids, (0, self.max_text_length - text_ids.size(-1)))
        #print("填充之后的text_ids", text_ids)
        attention_mask = F.pad(attention_mask, (0, self.max_text_length - attention_mask.size(-1)))
        token_type_ids = F.pad(token_type_ids, (0, self.max_text_length - token_type_ids.size(-1)))
        label_tensor = torch.tensor(label_mapped, dtype=torch.long)

        return text_ids, attention_mask, token_type_ids, image,  label_tensor

image_transform = get_image_transforms()# 图像转换
train_custom_dataset = CustomDataset(train_data_processed, image_transform=image_transform)
val_custom_dataset = CustomDataset(val_data_processed, image_transform=image_transform)
test_custom_dataset = CustomDataset(test_data_processed, image_transform=image_transform)

batch_size = 8

#获得按照batch加载的data
val_data_loader = DataLoader(val_custom_dataset, batch_size=batch_size, shuffle=True)
train_data_loader = DataLoader(train_custom_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_custom_dataset, batch_size=batch_size, shuffle=False)

hidden_size = 128
num_classes = 3

if config.fuse_model_type == 'SimpleMultimodel':
    from models.SimpleMultimodel import Model
elif config.fuse_model_type == 'ImprovedMultimodel':
    print("using ImprovedMultimodel")
    from models.ImprovedMultimodel import Model

model = Model(hidden_size, num_classes)
weights = [1.0, 2.0, 6.0]
class_weights = torch.FloatTensor(weights)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)#交叉熵
optimizer = Adam(model.parameters(), lr=0.00001)#优化器

#确保模型的可复现性
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def train(train_data_loader, model_name):
    print("begin training")

    torch.manual_seed(42)

    loss_df = pd.DataFrame(columns=['Epoch', 'Iteration', 'Loss'])
    metrics_df = pd.DataFrame(columns=['Epoch', 'Iteration', 'Accuracy', 'Precision', 'Recall', 'F1'])
    for epoch in range(config.epoch):
        model.train()#训练模式
        total_loss = 0.0
        num_batches = 0

        for batch in train_data_loader:
            text_ids, attention_mask, token_type_ids, image, label = batch
            optimizer.zero_grad()#清零梯度
            model_output = model(text_ids, attention_mask, token_type_ids, image)#前向运算
            _, predicted_labels = torch.max(model_output, 1)
            print("predicted_labels", predicted_labels)
            accuracy = accuracy_score(predicted_labels, label)
            print("Accuracy", accuracy)
            loss = criterion(model_output, label)#计算损失函数
            print(f'Epoch [{epoch + 1}/{config.epoch}], Loss: {loss.item()}, Batch: {num_batches}')
            loss.backward()#反向传播
            optimizer.step()#更新权重

            total_loss += loss.item()
            num_batches += 1

            loss_df = pd.concat([loss_df, pd.DataFrame({'Epoch': [epoch+1], 'Iteration': [num_batches], 'Loss': [loss.item()]})])

        model.eval()
        with torch.no_grad():
            for valbatch in val_data_loader:
                text_ids, attention_mask, token_type_ids, image, label = valbatch
                model_output = model(text_ids, attention_mask, token_type_ids, image)  # 前向传播
                _, predicted_labels = torch.max(model_output, 1)
                print("predicted_labels", predicted_labels)
                accuracy = accuracy_score(label, predicted_labels)
                precision = precision_score(label, predicted_labels, average='macro')
                recall = recall_score(label, predicted_labels, average='macro')
                f1 = f1_score(label, predicted_labels, average='macro')
                metrics_df = pd.concat([metrics_df, pd.DataFrame({'Epoch': [epoch+1], 'Iteration': [num_batches], 'Accuracy': [accuracy], 'Precision': [precision], 'Recall':[recall], 'F1':[f1]})])

                print(f'Epoch [{epoch + 1}/{config.epoch}], Accuracy: {accuracy}, Precision: {precision}, Recall:{recall}, F1:{f1}')

        average_loss = total_loss / num_batches
        print(f'Epoch [{epoch + 1}/{config.epoch}], Loss: {average_loss:.4f}')

    metric_os = current_dir + '\Metrics' + f'\{model_name}_metrics.csv'
    loss_os = current_dir + '\Lossdata' + f'\{model_name}_loss.csv'
    model_os = r'.\Savemodel' + f'\{config.fuse_model_type}.pth'

    loss_df.to_csv(loss_os, index=False)
    metrics_df.to_csv(metric_os, index=False)
    torch.save(model.state_dict(), model_os)
    return

def val_test(val_data_loader):
    model_os = r'.\Savemodel' + f'\{config.fuse_model_type}.pth'
    model.load_state_dict(torch.load(model_os))
    i = 0

    model.eval()
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    for data in val_data_loader:
        text_ids, attention_mask, token_type_ids, image, label = data
        #print("text_ids", text_ids)
        #print("img", image)
        model_output = model(text_ids, attention_mask, token_type_ids, image)  # 前向传播
        _, predicted_labels = torch.max(model_output, 1)
        print("predicted_labels", predicted_labels)

        accuracy = accuracy_score(label, predicted_labels)
        precision = precision_score(label, predicted_labels, average='macro')
        recall = recall_score(label, predicted_labels, average='macro')
        f1 = f1_score(label, predicted_labels, average='macro')

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        i = i + 1
        print(f'Batch:{i}, Accuracy: {accuracy}, Precision: {precision}, Recall:{recall}, F1:{f1}')

    accuracy_cross = np.array(accuracy_list).mean()
    precision_cross = np.array(precision_list).mean()
    recall_cross = np.array(recall_list).mean()
    f1_cross = np.array(f1_list).mean()
    print(f'Accuracy_corss: {accuracy_cross}, Precision_cross: {precision_cross}, Recall_cross:{recall_cross}, F1_cross:{f1_cross}')
    return

def test():
    model_os = r'.\Savemodel' + f'\{config.fuse_model_type}.pth'
    model.load_state_dict(torch.load(model_os))
    model.eval()
    all_predictions = []

    for data in test_data_loader:
        text_ids, attention_mask, token_type_ids, image, label = data
        model_output = model(text_ids, attention_mask, token_type_ids, image)  # 前向传播
        _, predicted_labels = torch.max(model_output, 1)
        print("predicted_labels", predicted_labels)
        all_predictions.extend(predicted_labels.tolist())
        print("all_predictions", all_predictions)

    print("all_predictions", all_predictions)
    int2str = {0: 'positive', 1: 'negative', 2: 'neutral'}
    all_predictions_char = [int2str[label] for label in all_predictions]

    """
    output_file_path = current_dir + '/result.txt'
    input_file_path = current_dir + '/dataset/test_without_label.txt'

    with open(input_file_path, 'r') as input_file:
        lines = input_file.readlines()
    with open(output_file_path, 'w') as output_file:
        output_file.write(lines[0])  # 写入第一行（guid, tag）

        # 替换文件中的标签并写入新文件
        for i in range(1, len(lines)):
            output_file.write(f"{lines[i].strip(',')}, {all_predictions_char[i - 1]}\n")
    """
    print("all_predictions_char", all_predictions_char)
    return all_predictions_char

if __name__ == "__main__":
    all_predictions_char = []
    if args.do_train:
        train(train_data_loader, config.fuse_model_type)
    elif args.do_val_test:
        val_test(val_data_loader)
    elif args.do_test:
        all_predictions_char=test()

    """
    i = 0
    for data in test_data_loader:
        text_ids, attention_mask, token_type_ids, image, label_tensor = data
        i = i + 1
        print("i", i)
        print("text_ids", text_ids)
        print("attention_mask", attention_mask)
        print("token_type_ids", token_type_ids)
        print("image", image)
        print("label_tensor", label_tensor)
    """