from typing import List, Optional
from transformers import BertForTokenClassification,BertTokenizer,BertConfig
from CRF_Model import CRF
import torch
import torch.nn as nn
import os
import code

MODEL_NAME = "bert-base-chinese-model.bin"
CONFIG_NAME = "bert-base-chinese-config.json"
VOB_NAME = "bert-base-chinese-vocab.txt"

class BertCrf(nn.Module):
    def __init__(self,config_name:str,model_name:str = None,num_tags: int = 2, batch_first:bool = True) -> None:
        self.batch_first = batch_first

        if not os.path.exists(config_name):
            raise ValueError(
                "未找到模型配置文件 '{}'".format(config_name)
            )
        else:
            self.config_name = config_name

        if model_name is not None:
            if not os.path.exists(model_name):
                raise ValueError(
                    "未找到模型预训练参数文件 '{}'".format(model_name)
                )
            else:
                self.model_name = model_name
        else:
            self.model_name = None

        super().__init__()
        self.bert_config = BertConfig.from_pretrained(self.config_name)
        self.bert_config.num_labels = num_tags
        self.model_kwargs = {'config': self.bert_config}

        if self.model_name is not None:
            self.bertModel = BertForTokenClassification.from_pretrained(self.model_name, **self.model_kwargs)
        else:
            self.bertModel = BertForTokenClassification(self.bert_config) 

        self.crf_model = CRF(num_tags=num_tags,batch_first=batch_first)
        
    def forward(self,input_ids:torch.Tensor,
                tags:torch.Tensor = None,
                attention_mask:Optional[torch.ByteTensor] = None,
                token_type_ids=torch.Tensor,
                decode:bool = True,      
                reduction: str = "mean") -> List:

        emissions = self.bertModel(input_ids = input_ids,attention_mask = attention_mask,token_type_ids=token_type_ids)[0]
        
        # 这里在seq_len的维度上去头，是去掉了[CLS]，去尾巴有两种情况
        # 1、是 <pad> 2、[SEP]

        new_emissions = emissions[:,1:-1] 
        new_mask = attention_mask[:,2:] 

        if tags is None:
            loss = None
            pass
        else:
            new_tags = tags[:, 1:-1] #
            loss = self.crf_model(emissions=new_emissions, tags=new_tags, mask=new_mask, reduction=reduction) # [1]
            
            
        if decode:
            tag_list = self.crf_model.decode(emissions = new_emissions,mask = new_mask)
            return [loss, tag_list]


        return [loss]




