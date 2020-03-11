# KBQA-BERT
## 基于知识图谱的问答系统，BERT做命名实体识别和句子相似度

## Introduction
本项目主要由两个重要的点组成，一是**基于BERT的命名实体识别**，二是**基于BERT的句子相似度计算**，本项目将这两个模块进行融合，构建基于BERT的KBQA问答系统.[**论文传送门**!](http://www.cnki.com.cn/Article/CJFDTotal-DLXZ201705041.htm) 
详细介绍，请看我的博客：https://blog.csdn.net/weixin_46133588/article/details/104700425

### 环境配置

    Python版本为3.6
    pytorch版本为1.1.0
    windows10
    数据在Data中，更多的数据在[**NLPCC2016**](http://tcci.ccf.org.cn/conference/2016/pages/page05_evadata.html) 和    [**NLPCC2017**](http://tcci.ccf.org.cn/conference/2017/taskdata.php)。    
    
### 目录说明

    Input/data/文件夹存放原始数据和处理好的数据
        1-split_data.py  切分数据
        2-construct_dataset.py  生成NER_Data的数据
        4-print-seq-len.py 查看数据长度
        construct_dataset_attribute.py  生成Sim_Data的数据
        triple_clean.py  生成三元组数据
        load_dbdata.py  将数据导入mysql db
    config文件夹需要下载BERT的中文配置文件：bert-base-chinese-config.json bert-base-chinese-model.bin bert-base-chinese-vocab.txt
    NLPC2016KBQA存放原始数据的地方
    DB_Data/   ner_data/  sim_data/ 为数据输出存放的文件夹

    output/模型输出的目录
    
    基于BERT的命名实体识别模块
    - BERT_CRF.py
    - CRF_Model.py
    - NER_main.py
    - test_NER.py
    
    基于BERT的句子相似度计算模块
    - SIM_main.py
    - test_SIM.py

    KBQA模块
    - test_pro.py
    
 ### 使用说明
    
    - python NER_main.py --data_dir ./input/data/ner_data --vob_file ./input/config/bert-base-chinese-vocab.txt  --model_config  ./input/config/bert-base-chinese-config.json  --output  ./output --max_seq_length 64 --do_train --train_batch_size 12 --eval_batch_size 16 --gradient_accumulation_steps 4 --num_train_epochs 15
    NER训练和调参
    
    
    - python SIM_main.py --data_dir ./input/data/sim_data --vob_file ./input/config/bert-base-chinese-vocab.txt  --model_config  ./input/config/bert-base-chinese-config.json  --output  ./output --max_seq_length 64 --do_train --train_batch_size 12 --eval_batch_size 16 --gradient_accumulation_steps 4 --num_train_epochs 5 --pre_train_model ./input/config/bert-base-chinese-model.bin
    相似度训练阶段
    
    - python test_pro.py
    基于KB的问答测试
  
 --------------------------------------------------------------
**如果觉得我的工作对您有帮助，请不要吝啬右上角的小星星哦！欢迎Fork和Star！也欢迎一起建设这个项目！**    
**有时间就会更新问答相关项目，有兴趣的同学可以follow一下**  
**留言请在Issues或者email 492070779@qq.com**
    
