
### What can we learn in this class that transfers to frontier model？
There are three types of knowledge:
Mechanics: how things work (what a Transformer is, how model parallelism works)
Mindset: squeezing the most out of the hardware, taking scaling seriously
Intuitions: which data and modeling decisions yield good accuracy

### The bitter lesson
Wrong interpretation: scale is all that matters, algorithms don't matter.
Right interpretation: algorithms that scale are what matter.

### accuracy = efficiency × resources

In fact, efficiency is way more important at larger scales (can't afford to be wasteful).
[Hernandez+ 2020] showed 44x algorithmic efficiency on ImageNet between 2012 and 2019.

Framing: what is the best model one can build given a certain compute and data budget?

### In other words, maximize efficiency!

### 1、Tokenization
What are the atoms（最小基本单元） that the model operates on?
Formally: a tokenizer converts between raw inputs (bytes) and sequences of integers (tokens)

<img width="2395" height="983" alt="Image" src="https://github.com/user-attachments/assets/0673caa3-6303-45de-af6f-f2f209c65c5b" />

Popular tokenizer: Byte-Pair Encoding (BPE) https://arxiv.org/abs/1508.07909 
Intuition: break input into frequently-occuring chunks
   
#### Efficiency lens：
（1）Reduce context length (1000 bytes → ~250 tokens)
（2）Adaptive computation (more modeling capacity on interesting parts of input)
   
#### The dream: 
tokenizer-free model architectures, which operate directly on bytes （https://arxiv.org/abs/2105.13626；https://arxiv.org/abs/2412.09871； https://arxiv.org/abs/2406.19223；https://arxiv.org/abs/2507.07955）
These are promising, but have not yet been scaled up to the frontier.

Raw text is generally represented as **Unicode strings**.

例如：string = "Hello, 🌍! 你好!"
 
A language model places a probability distribution over sequences of tokens (usually represented by integer indices).

例如：indices = [15496, 11, 995, 0]

So we need a procedure that **encodes strings into tokens**.
We also need a procedure that **decodes tokens back into strings**.
A Tokenizer is a class that **implements the encode and decode methods**.

To get a feel for how tokenizers work, play with this：https://tiktokenizer.vercel.app/?encoder=gpt2

### Observations
（1）A word and its preceding space are part of the same token (e.g., " world").  
（2）A word at the beginning and in the middle are represented differently (e.g., "hello hello").  
（3）Numbers are tokenized into every few digits.

### Compression ratio: number of bytes per token
即 **压缩比 compression_ratio = 原始文本UTF-8总字节数 / token数量**
The larger the compression ratio, the shorter the sequence (good since attention is quadratic in sequence length)，即：压缩比越大，token序列越短.
One could increase compression ratio by **increasing vocabulary size (number of possible token values increases)**, leading to sparsity（稀疏性，即许多token在数据中极少出现，表征训练不足）.

## 如何表征token？

There are approximately 150K Unicode characters. 
#### Problem 1: this is a very large vocabulary.
#### Problem 2: many characters are quite rare (e.g., 🌍), which is inefficient use of the vocabulary.
This tokenizer is the worst of both worlds (large vocabulary, low compression ratio).

换一种方法：（转为字节）
Unicode strings can be represented as a sequence of bytes, which can be represented by integers between 0 and 255.

The vocabulary is nice and small: a byte can represent 256 values.
vocabulary_size = 256  

### Problem：   
What about the compression rate?
The compression ratio is terrible, which means the sequences will be too long.
Given that the context length of a Transformer is limited (since attention is quadratic), this is not looking great...

其他方法：
Another approach (closer to what was done classically in NLP) is to split strings into words.

What's good: each token is meaningful (since humans invented words).  

Problem：
（1）Many words are rare and the model won't learn much about them.
（2）This doesn't obviously provide a fixed vocabulary size.
（3）New words we haven't seen during training get a special UNK token, which is ugly and can mess up perplexity calculations.

# Byte Pair Encoding（BPE）
Intuition: common sequences of bytes are represented by a single token, rare sequences are represented by many tokens.
Sketch: start with each byte as a token, and successively merge the most common pair of adjacent tokens.

### Example：

原始文本：`the cat in the hat`

### Step 0 初始化
把文本拆分为最小单元（字符，模拟底层UTF-8字节）
序列：`t h e   c a t   i n   t h e   h a t`
基础 vocab：`{t, h, e,  , c, a, i, n}`

### Step 1 统计相邻配对频次，选出最高频 `t h`
合并 `t + h → th`
新序列：`th e   c a t   i n   th e   h a t`
vocab 新增：`th`

### Step 2 重新统计配对，最高频 `th e`
合并 `th + e → the`
新序列：`the   c a t   i n   the   h a t`
vocab 新增：`the`

### Step 3 重新统计配对，最高频 `a t`
合并 `a + t → at`
最终序列：`the   c at   i n   the   h at`
vocab 新增：`at`

结果：
1. 高频片段 `the`、`at` 合并为**单个token**；
2. 极低频次片段无法合并，会保留大量基础单元（多个token）。

start with each byte as a token, and successively merge the most common pair of adjacent tokens.
从最小单元起步，循环统计相邻token对频率，不断合并出现最多的配对，直到达到预设词表上限。

