用于存放数据相关的文件和文件夹。

raw/

phoenix14t/

存放原始的 PHOENIX14T 数据集文件，例如 train.sentences.txt、dev.sentences.txt 和 test.sentences.txt。

processed/

存放数据预处理后的文件，例如提取的词汇表、索引映射等。

embeddings/

存放预训练的词嵌入模型，例如 cc.de.300.bin。

```
conda env create -f environment.yaml
```
