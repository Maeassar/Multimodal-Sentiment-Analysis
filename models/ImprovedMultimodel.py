import torch
import torch.nn as nn
from transformers import BertModel
from torchvision.models import resnet50

class Model(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(Model, self).__init__()

        self.bert_model = BertModel.from_pretrained("./bert_multilingual") # BERT分支, 用于处理文字
        self.text_trans = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(768, hidden_size),
            nn.ReLU(inplace=True)
        )

        self.resnet_model = resnet50(pretrained=True) # ResNet分支， 用于处理图片

        self.resnet_fc = nn.Linear(1000, hidden_size)
        self.resnet_relu = nn.ReLU(inplace=True)

        self.attention = nn.TransformerEncoderLayer(
            d_model=hidden_size * 2,
            nhead=8,
            dropout=0.5
        )

        self.output = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, text_ids, attention_mask, token_type_ids, image_input):
        assert text_ids.shape == attention_mask.shape, "error! text_ids and attention_mask must have same shape!"

        text_output = self.bert_model(input_ids=text_ids,
                                      token_type_ids=token_type_ids,
                                      attention_mask=attention_mask)["pooler_output"]
        text_output = self.text_trans(text_output)
        image_output = self.resnet_model(image_input)
        image_output = self.resnet_fc(image_output)

        attention_output = self.attention(
            torch.cat((text_output, image_output), dim=1)
        )

        output = self.output(attention_output)# 输出层

        return output
