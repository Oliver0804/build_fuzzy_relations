# src/utils/helpers.py

import numpy as np
from gensim.models import KeyedVectors
from tqdm import tqdm

# 导入 FastText
from gensim.models import FastText

def load_embeddings(embedding_model_path):
    """
    加载预训练的 FastText 词嵌入模型。

    :param embedding_model_path: FastText 模型文件路径（.bin 文件）
    :return: 词嵌入模型对象
    """
    embedding_model = FastText.load_fasttext_format(embedding_model_path)
    return embedding_model.wv  # 返回词向量部分

def compute_similarity_matrix(vocab, embedding_model):
    """
    计算词汇之间的余弦相似度矩阵。

    :param vocab: 词汇表列表
    :param embedding_model: 词嵌入模型
    :return: 相似度矩阵，OOV 词汇集合
    """
    vocab_size = len(vocab)
    similarity_matrix = np.zeros((vocab_size, vocab_size))
    oov_words = set()
    word_vectors = {}
    for i in tqdm(range(vocab_size), desc="加载词向量"):
        word = vocab[i]
        if word in embedding_model:
            word_vectors[word] = embedding_model[word]
        else:
            oov_words.add(word)
    for i in tqdm(range(vocab_size), desc="计算相似度矩阵"):
        word_i = vocab[i]
        if word_i in oov_words:
            continue
        vector_i = word_vectors[word_i]
        for j in range(i+1, vocab_size):
            word_j = vocab[j]
            if word_j in oov_words:
                continue
            vector_j = word_vectors[word_j]
            # 计算余弦相似度
            similarity = np.dot(vector_i, vector_j) / (np.linalg.norm(vector_i) * np.linalg.norm(vector_j))
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # 对称矩阵
    return similarity_matrix, oov_words
