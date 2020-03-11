from typing import List, Optional
import torch
import code
import torch.nn as nn


class CRF(nn.Module):
    def __init__(self,num_tags : int = 2, batch_first:bool = True) -> None:
        # if num_tags <= 0:
        #     raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags # 3
        self.batch_first = batch_first # true
        # start 到其他tag(不包含end)的得分
        self.start_transitions = nn.Parameter(torch.empty(num_tags)) # tensor([-0.0025, -0.0600, -0.0841], requires_grad=True)
        # 到其他tag(不包含start)到end的得分
        self.end_transitions = nn.Parameter(torch.empty(num_tags)) # tensor([-0.0673,  0.0638, -0.0416], requires_grad=True)
        # 从 _compute_normalizer 中 next_score = broadcast_score + self.transitions + broadcast_emissions 可以看出
        # transitions[i][j] 表示从第i个tag 到第j个 tag的分数
        self.transitions = nn.Parameter(torch.empty(num_tags,num_tags)) # [3,3] tensor([[1,1,1],[1,1,1],[1,1,1]])

        self.reset_parameters()
        # 

    def reset_parameters(self):
        init_range = 0.1
        nn.init.uniform_(self.start_transitions,-init_range,init_range)
        nn.init.uniform_(self.end_transitions,-init_range,init_range)
        nn.init.uniform_(self.transitions, -init_range, init_range)


    def forward(self, emissions:torch.Tensor,
                tags:torch.Tensor = None,
                mask:Optional[torch.ByteTensor] = None,
                reduction: str = "mean") -> torch.Tensor:

        # emissions.shape = [12,62,3]=[batchsize, seq_len-2, num_tag]
        mask=torch.tensor(mask,dtype=torch.uint8) # [32,62]
        self._validate(emissions, tags = tags ,mask = mask)
        

        reduction = reduction.lower()

        if mask is None:
            mask = torch.ones_like(tags,dtype = torch.uint8)

        if self.batch_first:
            # emissions.shape (seq_len,batch_size,tag_num)
            emissions = emissions.transpose(0,1) # [62,32,3]
            tags = tags.transpose(0,1) # [62,32]
            mask = mask.transpose(0,1) # [62,32]

        # numerator shape: (batch_size,) [32]
        numerator = self._computer_score(emissions=emissions,tags=tags,mask=mask) # 计算s(X,y)
        
        # shape: (batch_size,) [32]
        denominator = self._compute_normalizer(emissions=emissions,mask=mask)  # 计算 log(Σe^(x,y'))
        
        # loss function
        # log(Σe^(S(X,y))) - S(X，y)
        llh = denominator - numerator # [batch_size] 32
        

        if reduction == 'none':
            return llh
        elif reduction == 'sum':
            return llh.sum()
        elif reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'

        return llh.sum() / mask.float().sum()

    def decode(self,emissions:torch.Tensor,
               mask : Optional[torch.ByteTensor] = None) ->List[List[int]]:
        self._validate(emissions=emissions,mask=mask)

        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2],dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0,1) # [62,32,3]
            mask = mask.transpose(0,1) # [62,32]

        return self._viterbi_decode(emissions,mask)

    # 判断有效性
    def _validate(self,
                  emissions:torch.Tensor,
                  tags:Optional[torch.LongTensor] = None ,
                  mask:Optional[torch.ByteTensor] = None) -> None:

        if tags is not None:
            no_empty_seq = not self.batch_first and mask[0].all() #  false
            no_empty_seq_bf = self.batch_first and mask[:,0].all()


    # 计算公式s(X,y)
    def _computer_score(self,
                        emissions:torch.Tensor,
                        tags:torch.LongTensor,
                        mask:torch.ByteTensor) -> torch.Tensor:

        # batch second
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()
        tags.cuda()
        
        # 62        32
        seq_length,batch_size = tags.shape
        mask = mask.float().cuda()

        # self.start_transitions  start 到其他tag(不包含end)的得分
        
        score = self.start_transitions[tags[0]] # tag[0].shape = [32] 每一句的第一个单词，start到其它tag的得分，随机给一个值
        # code.interact(local = locals())

        score += emissions[0,torch.arange(batch_size),tags[0]] # 计算所有句子中第一个单词的发射的得分

        for i in range(1,seq_length): # [1,2,...,seq_length-1]
            # if mask[i].sum() == 0:
            #     break
            # transitions[i][j] 表示从第i个tag 到第j个tag的分数
            score += self.transitions[tags[i-1], tags[i]] * mask[i] # Aij

            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]    # P{i,y_j}

        # 这里是为了获取每一个样本最后一个词的tag。
        # shape: (batch_size,)   每一个batch 的真实长度
        # .long 变成整型 .sum(dim=0) 计算每个句子中一共有多少个字
        seq_ends = mask.long().sum(dim=0) - 1
        
        # 每个样本最后一个词的tag
        last_tags = tags[seq_ends,torch.arange(batch_size)]

        # shape: (batch_size,) 每一个样本到最后一个词的得分加上之前的score
        score += self.end_transitions[last_tags]

        return score

    # 计算 log(Σe^(S(X,y)))
    def _compute_normalizer(self,
                            emissions:torch.Tensor ,
                            mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags) [62,32,3]
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()
        mask = mask.cuda()

        seq_length = emissions.size(0)
        
        score = self.start_transitions + emissions[0] # [3] + [32,3]  每个句子的第一个单词，对应的3中不同tag的得分
        # code.interact(local = locals())

        for i in range(1,seq_length):

            # shape : (batch_size,num_tag,1) [32,3,1]
            # <start> -> 别的tag的得分
            broadcast_score = score.unsqueeze(dim=2) # 初始化得分

            # shape: (batch_size,1,num_tags)
            # 每个句子的第i个单词的发射分数 [32,1,3]
            broadcast_emissions = emissions[i].unsqueeze(1)

            # 广播得分 + transitions[i][j] 表示从第j个tag 到第 i 个 tag的分数+ 每句第i个单词的发射得分  [32,3,3]
            next_score = broadcast_score + self.transitions + broadcast_emissions

            next_score = torch.logsumexp(next_score,dim = 1) # [32,3]
         
            # 依次取出每句话的i个单词的mask, 有的话，有得分，没有的话，就原来的值 [32,3]
            score = torch.where(mask[i].unsqueeze(1),next_score,score)

        # shape (batch_size,num_tags)
        # 到其他tag(不包含start)到end的得分
        score += self.end_transitions

        # shape: (batch_size)
        return torch.logsumexp(score,dim=1)

    def _viterbi_decode(self,emissions : torch.FloatTensor ,
                        mask : torch.ByteTensor) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        mask=torch.tensor(mask,dtype=torch.uint8).cuda()
        assert mask[0].all()


        seq_length , batch_size = mask.shape
        # self.start_transitions  start 到其他tag(不包含end)的得分
        # <start>->其他tag的发射得分 + 每句话的第一个字的tag的发射得分
        # emissions.shape = [62,32,3]  start_transitions.shape = [3]
        score = self.start_transitions + emissions[0] # 广播
        history = []

        for i in range(1,seq_length):
            # score.shape = [32,3] -> [32,3,1]
            # 起始得分，发射得分, 扩展维度，公式中的expand previous
            broadcast_score = score.unsqueeze(2) 

            # emissions.shape = [62,32,3] 然后emissions[i]是每句话中的第i个单词的发射得分
            # emissions[i].shape = [32,3].unsqueeze(1) = [32,1,3]
            broadcast_emission = emissions[i].unsqueeze(1) # 扩展维度
 
            #            初始得分         + 发射得分         + 之前得分+现在得分
            # [32,3,3]=  [32,3,1]        + [3,3,3]          + [32,1,3]
            #            初始得分           公式中的t，转移得分  发射得分，单词wi->tagj的发射概率
            next_score = broadcast_score + self.transitions + broadcast_emission
            
            # 每句话中每个单词对应的tag的得分最大值，tag[i]->tag[j]最大得分
            # next_score.shape = [32,3] =indices.shape
            # 这个时刻中的最大值被保留下来
            next_score, indices = next_score.max(dim=1)

            # 不计算padding部分的得分
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

            history.append(indices)

        # 遍历完一句话，还得加上最后 <end> tag  [32,3]
        score += self.end_transitions

        # 计算到最后一个单词的下标
        seq_ends = mask.long().sum(dim=0) - 1  # [32]
        best_tags_list = []

        jj = 0
        for idx in range(batch_size): # 32 

            # score.shape = [32,3] 每句话中找最好的tag
            _,best_last_tag = score[idx].max(dim = 0) # 然后找最好的tag

            best_tags= [best_last_tag.item()]
            
            # history[:seq_ends[idx]].shape  (seq_ends[idx])

            # history 的长度是一个句子的长度 61
            # history[i].shape = [32,3]
            # history[:seq_ends[idx] 取句子长度，seq_ends之后是padding部分
            # reversed(history[:seq_ends[idx]]) 将句子反过来
            # hist第一个取到的是最后一个字
            for hist in reversed(history[:seq_ends[idx]]): # 画图
                # hist.shape = [32,3]
                # code.interact(local = locals())
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            best_tags.reverse()
            best_tags_list.append(best_tags)
        return best_tags_list