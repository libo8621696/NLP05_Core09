import os
import io
import utils

import torch
from torch import nn
import torch.optim as optim

from collections import Counter
import random
import numpy as np

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from torchnlp.download import download_files_maybe_extract
from torchnlp.encoders.text import DEFAULT_EOS_TOKEN
from torchnlp.encoders.text import DEFAULT_UNKNOWN_TOKEN


def penn_treebank_dataset(
    directory='data/penn-treebank',
    train=False,
    dev=False,
    test=False,
    train_filename='ptb.train.txt',
    dev_filename='ptb.valid.txt',
    test_filename='ptb.test.txt',
    check_files=['ptb.train.txt'],
    urls=[
        'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt',
        'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt',
        'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt'
    ],
    unknown_token=DEFAULT_UNKNOWN_TOKEN,
    eos_token=DEFAULT_EOS_TOKEN):

    download_files_maybe_extract(urls=urls, directory=directory, check_files=check_files)

    ret = []
    splits = [(train, train_filename), (dev, dev_filename), (test, test_filename)]
    splits = [f for (requested, f) in splits if requested]
    for filename in splits:
        full_path = os.path.join(directory, filename)
        text = []
        with io.open(full_path, encoding='utf-8') as f:
            for line in f:
                text.extend(line.replace('<unk>', unknown_token).split())
                text.append(eos_token)
        ret.append(text)

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)

## 通过以上的penn_treebank_dataset()导入ptb数据集并放入data/penn-treebank文件夹中
penn_treebank_dataset()

## 从data/penn-treebank文件夹中的ptb.train.txt读取数据
with open('./data/penn-treebank/ptb.train.txt') as f_ptb_train:
    text_ptb_train = f_ptb_train.read()

# 打印前50个字符做测试
# print(text_ptb_train[:100])

# 得到如下结果：
# aer banknote berlitz calloway centrust cluett fromstein gitano guterman hydro-quebec ipo kia memote
# 可以看出，这里的只是字符的输出，没有进行分词，因此在生成词向量前需要对文本进行预处理，这里的预处理代码都在utils.py之中，主要的作用是将一些分隔符转换为特殊符号，同时 删除出现不超过5次的单词，减少这些低频词所带来的数据噪声，提升词向量表示的质量，最终会返回词列表。

words = utils.preprocess(text_ptb_train)
# print(words[:30])

#打印出前30个词语如下 ['pierre', '<unk>', 'n', 'years', 'old', 'will', 'join', 'the', 'board', 'as', 'a', 'nonexecutive', 'director', 'nov', '<PERIOD>', 'n', 'mr', '<PERIOD>', '<unk>', 'is', 'chairman', 'of', '<unk>', 'n', '<PERIOD>', 'v', '<PERIOD>', 'the', 'dutch', 'publishing']

# print("Total words in text: {}".format(len(words)))
# print("Unique words: {}".format(len(set(words)))) 

# 通过`set` 去除重复的词，得到如下结果，训练集合文本共有905112个单词，除去重复的单词共有9515个单词。
# Total words in text: 905112 Unique words: 9515

# 在utils.py中，通过建立查找表的函数create_lookup_tables得到两个字典，一个字典是将词语映射到出现ID号码，另一个是将ID号码返回到词语本身。ID 号码的大小是根据出现频次大小来逆序排列的，例如the出现次数最多，则编号为0,出现次数次多的则编号为1，依此类推。
vocab_to_int, int_to_vocab = utils.create_lookup_tables(words)
int_words = [vocab_to_int[word] for word in words]

# print(int_words[:30])
# 对应于前30个单词的相应编号为[8510, 1, 2, 74, 394, 33, 2113, 0, 147, 19, 5, 8511, 276, 409, 8, 2, 23, 8, 1, 13, 142, 3, 1, 2, 8, 2460, 8, 0, 3047, 1584]

# 二次采样，对于“the”，“of”以及“for”这类过高频词，对上下文的信息贡献并不是很大，去除这些过高频词有助于提升训练速度。其公式如下所示P(w^{i}) = 1 - \sqrt{\frac{t}{f(w_{i})}}，其中t表示阈值参数，f表示单词频数，P表示该单词被去除的概率

threshold = 1e-5
word_counts = Counter(int_words)
# print(list(word_counts.items())[0])  

total_count = len(int_words)
freqs = {word: count/total_count for word, count in word_counts.items()}
p_drop = {word: 1 - np.sqrt(threshold/freqs[word]) for word in word_counts}
# 通过二次下采样公式去除相关的高频词，得到新的训练集词语列表
train_words = [word for word in int_words if random.random() < (1 - p_drop[word])]

# print(train_words[:30])
# 新的词语ID号码有所改变，如下所示，可以看出相关的the等过高频词已经去除 [8510, 19, 8511, 8, 3, 2460, 3047, 7185, 2461, 2154, 8511, 439, 3647, 3133, 4, 5848, 4204, 5849, 30, 4047, 3133, 7186, 6679, 114, 2641, 7771, 2462, 2973, 2575, 5848]
# print(len(Counter(train_words))) 

# 在使用skip-gram结构之前，需要明确传入网络的词语批次batch，需要定义窗口的大小以及窗口范围内的词语列表。


def get_target(words, idx, window_size=5):
    ''' Get a list of words in a window around an index. '''
    
    R = np.random.randint(1, window_size+1)
    start = idx - R if (idx - R) > 0 else 0
    stop = idx + R
    target_words = words[start:idx] + words[idx+1:stop+1]
    
    return list(target_words)

int_text = [i for i in range(10)]
print('Input: ', int_text)
idx=5 # 对词语进行测试，查看某个固定的单词附近，如果随机选取5个词语范围之内的窗口，是否可以得到附近的词语ID

target = get_target(int_text, idx=idx, window_size=5)
print('Target: ', target) 
# 通过Target我们进行了两侧尝试，一次得到的是[4,6]，这是窗口大小为1的情形，一次得到的是[3, 4, 6, 7]，这是窗口大小为3的情形。
# (TF2-Torch) libo@libos-MacBook-Pro lesson09 % python3 Core_lesson09.py
# Input:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Target:  [4, 6]
# (TF2-Torch) libo@libos-MacBook-Pro lesson09 % python3 Core_lesson09.py
# Input:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Target:  [3, 4, 6, 7]

# 获取批次数据batch data，利用以上的get_target函数恢复目标上下文的窗口内词语



def get_batches(words, batch_size, window_size=5):
    ''' Create a generator of word batches as a tuple (inputs, targets) '''
    
    n_batches = len(words)//batch_size
    
    # 全batch
    words = words[:n_batches*batch_size]
    
    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx+batch_size]
        for ii in range(len(batch)):
            batch_x = batch[ii]
            batch_y = get_target(batch, ii, window_size)
            y.extend(batch_y)
            x.extend([batch_x]*len(batch_y))
        yield x, y

int_text = [i for i in range(20)]
x,y = next(get_batches(int_text, batch_size=4, window_size=5))

# 通过cosine_similarity计算词向量的相似度

def cosine_similarity(embedding, valid_size=16, valid_window=100, device='cpu'):
    """ Returns the cosine similarity of validation words with words in the embedding matrix.
        Here, embedding should be a PyTorch embedding module.
    """
    
    
    # sim = (a . b) / |a||b|

    #这里随机抽取一些单词，通过cosine similarity计算与这些单词关系最近的单词
    
    embed_vectors = embedding.weight
    
    # 嵌入向量的大小 |b|
    magnitudes = embed_vectors.pow(2).sum(dim=1).sqrt().unsqueeze(0)
    
 
    # 从(0, window) 和（1000， 1000+window）范围内随机挑选N个单词
    valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
    valid_examples = np.append(valid_examples,random.sample(range(1000,1000+valid_window), valid_size//2))
    valid_examples = torch.LongTensor(valid_examples).to(device)
    
    valid_vectors = embedding(valid_examples)
    similarities = torch.mm(valid_vectors, embed_vectors.t())/magnitudes
        
    return valid_examples, similarities

class SkipGramNeg(nn.Module):
    def __init__(self, n_vocab, n_embed, noise_dist=None):
        super().__init__()
        
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.noise_dist = noise_dist
        
        # 为输入和输出单词定义嵌入层
        self.in_embed = nn.Embedding(n_vocab,n_embed)
        self.out_embed = nn.Embedding(n_vocab,n_embed)
        

        # 按照均匀分布对嵌入表格进行初始化
        
    def forward_input(self, input_words):
        # 返回输入词嵌入向量
        input_vectors = self.in_embed(input_words)
        return input_vectors
    
    def forward_output(self, output_words):
        # 返回输出词嵌入向量
        output_vectors = self.out_embed(output_words)
        return output_vectors
    
    def forward_noise(self, batch_size, n_samples):
        """ Generate noise vectors with shape (batch_size, n_samples, n_embed)"""
        if self.noise_dist is None:
            # 平均采样单词
            noise_dist = torch.ones(self.n_vocab)
        else:
            noise_dist = self.noise_dist
            
        # 从噪声分布中采样单词

        noise_words = torch.multinomial(noise_dist,batch_size * n_samples, replacement=True)
        
        device = "cuda" if model.out_embed.weight.is_cuda else "cpu"
        noise_words = noise_words.to(device)
        
        
        noise_vectors = self.out_embed(noise_words).view(batch_size,n_samples,self.n_embed)
        
        return noise_vectors

class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_vectors, output_vectors, noise_vectors):
        
        batch_size, embed_size = input_vectors.shape
        
        #  输入的词向量需要是完整词的列向量
        input_vectors = input_vectors.view(batch_size, embed_size, 1)
        
        # 输出的词向量需要是完整词的行向量
        output_vectors = output_vectors.view(batch_size, 1, embed_size)
        
        # bmm = batch matrix multiplication
        #  正确的log-sigmoid损失函数
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log()
        out_loss = out_loss.squeeze()
        
        # 不正确的log-sigmoid损失函数
        noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()
        noise_loss = noise_loss.squeeze().sum(1)  # 在噪声向量样本上将损失值相加

        # 获取平均的batch损失函数
        return -(out_loss + noise_loss).mean()


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 获取噪声分布

word_freqs = np.array(sorted(freqs.values(), reverse=True))
unigram_dist = word_freqs/word_freqs.sum()
noise_dist = torch.from_numpy(unigram_dist**(0.75)/np.sum(unigram_dist**(0.75)))

# 模型初始化
embedding_dim = 300
model = SkipGramNeg(len(vocab_to_int), embedding_dim, noise_dist=noise_dist).to(device)

# 使用定义好的损失函数
criterion = NegativeSamplingLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.003)

print_every = 1500
steps = 0
epochs = 80

# train for some number of epochs
for e in range(epochs):
    # get our input, target batches
    for input_words, target_words in get_batches(train_words, 512):
        steps += 1
        inputs, targets = torch.LongTensor(input_words), torch.LongTensor(target_words)
        inputs, targets = inputs.to(device), targets.to(device)
        
        # input, outpt, and noise vectors
        input_vectors = model.forward_input(inputs)
        output_vectors = model.forward_output(targets)
        noise_vectors = model.forward_noise(inputs.shape[0], 5)

        # negative sampling loss
        loss = criterion(input_vectors, output_vectors, noise_vectors)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # loss stats
        if steps % print_every == 0:
            print("Epoch: {}/{}".format(e+1, epochs))
            print("Loss: ", loss.item()) # avg batch loss at this point in training
            valid_examples, valid_similarities = cosine_similarity(model.in_embed, device=device)
            _, closest_idxs = valid_similarities.topk(6)

            valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
            for ii, valid_idx in enumerate(valid_examples):
                closest_words = [int_to_vocab[idx.item()] for idx in closest_idxs[ii]][1:]
                print(int_to_vocab[valid_idx.item()] + " | " + ', '.join(closest_words))
            print("...\n")

embeddings = model.in_embed.weight.to('cpu').data.numpy()

print(embeddings[:5])

viz_words = 80
tsne = TSNE()
embed_tsne = tsne.fit_transform(embeddings[:viz_words, :])

fig, ax = plt.subplots(figsize=(16, 16))
for idx in range(viz_words):
    plt.scatter(*embed_tsne[idx, :],  color='steelblue')
    plt.annotate(int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)

plt.show()

