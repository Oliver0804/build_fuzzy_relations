# src/data_preparation/prepare_data.py

import os
import csv

def load_phoenix14t_data(data_path):
    """
    加载 PHOENIX-2014-T 数据集，并提取德语句子列表。
    CSV 文件的列名为：
    name|video|start|end|speaker|orth|translation
    """
    german_sentences = []
    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        header = next(reader)  # 跳过表头
        for row in reader:
            if len(row) >= 7:
                translation = row[6].strip()
                german_sentences.append(translation)
    return german_sentences

def build_vocab(sentences):
    """
    构建词汇表和索引映射。

    :param sentences: 德语句子列表
    :return: 词汇表列表，word2idx，idx2word
    """
    vocab = set()
    for sentence in sentences:
        words = sentence.strip().split()
        vocab.update(words)
    vocab = sorted(list(vocab))
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return vocab, word2idx, idx2word
