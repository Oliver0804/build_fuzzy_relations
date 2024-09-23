# src/build_fuzzy_relations.py

import sys
import os
import numpy as np
import pickle

# 获取项目根目录并添加到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from collections import defaultdict
from tqdm import tqdm

# 导入自定义模块
from data_preparation.prepare_data import load_phoenix14t_data, build_vocab
from utils.helpers import load_embeddings, compute_similarity_matrix

def build_fuzzy_relations(similarity_matrix, word2idx, idx2word, similarity_threshold=0.7):
    """
    根据相似度矩阵构建模糊关系映射。

    :param similarity_matrix: 词汇之间的相似度矩阵
    :param word2idx: 词汇到索引的映射
    :param idx2word: 索引到词汇的映射
    :param similarity_threshold: 相似度阈值
    :return: 模糊关系映射字典
    """
    fuzzy_relations = defaultdict(dict)
    vocab_size = len(word2idx)

    print("构建模糊关系映射...")
    for i in tqdm(range(vocab_size), desc="构建模糊关系映射"):
        word_i_index = i
        related_words = {}
        for j in range(vocab_size):
            if i == j:
                continue
            similarity = similarity_matrix[i, j]
            if similarity > similarity_threshold:
                word_j_index = j
                related_words[word_j_index] = float(similarity)  # 隶属度设置为相似度
        if related_words:
            fuzzy_relations[word_i_index] = related_words

    print(f"具有模糊关系的词汇数量：{len(fuzzy_relations)}")
    return fuzzy_relations



def main():
    # 设置数据路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data', 'raw', 'phoenix14t', 'PHOENIX-2014-T.train.corpus.csv')
    embedding_model_path = os.path.join(project_root, 'data', 'embeddings', 'cc.de.300.bin')
    output_path = os.path.join(project_root, 'data', 'processed', 'fuzzy_relations.pkl')

    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"数据文件未找到：{data_path}")
        return
    if not os.path.exists(embedding_model_path):
        print(f"词嵌入模型未找到：{embedding_model_path}")
        return
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    # 加载数据并构建词汇表
    print("加载数据并构建词汇表...")
    german_sentences = load_phoenix14t_data(data_path)
    german_vocab, word2idx, idx2word = build_vocab(german_sentences)

    # 加载词嵌入模型
    print("加载预训练的德语词嵌入模型...")
    embedding_model = load_embeddings(embedding_model_path)

    # 计算相似度矩阵
    print("计算词汇之间的相似度...")
    similarity_matrix, oov_words = compute_similarity_matrix(german_vocab, embedding_model)

    print(f"OOV（未登录）词汇数量：{len(oov_words)}")

    # 构建模糊关系映射
    fuzzy_relations = build_fuzzy_relations(similarity_matrix, word2idx, idx2word, similarity_threshold=0.7)

    # 在构建词汇表后，保存 word2idx 和 idx2word
    vocab_data = {'word2idx': word2idx, 'idx2word': idx2word}
    vocab_path = os.path.join(project_root, 'data', 'processed', 'vocab.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab_data, f)
    print(f"词汇映射已保存到 {vocab_path}")
    
    # 保存模糊关系映射
    with open(output_path, 'wb') as f:
        pickle.dump(fuzzy_relations, f)
    print(f"模糊关系映射已保存到 {output_path}")

if __name__ == '__main__':
    main()
