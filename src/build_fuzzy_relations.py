# src/build_fuzzy_relations.py
import sys
import os
import numpy as np
import pickle
from datetime import datetime
from tqdm import tqdm

# 獲取專案根目錄並添加到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from collections import defaultdict

# 導入自訂模組
from data_preparation.prepare_data import load_phoenix14t_data, build_vocab
from utils.helpers import load_embeddings, compute_similarity_matrix

# 顏色代碼
class TerminalColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def build_fuzzy_relations(similarity_matrix, word2idx, idx2word, similarity_threshold=0.7):
    """
    根據相似度矩陣構建模糊關係映射。

    :param similarity_matrix: 詞彙之間的相似度矩陣
    :param word2idx: 詞彙到索引的映射
    :param idx2word: 索引到詞彙的映射
    :param similarity_threshold: 相似度閾值
    :return: 模糊關係映射字典
    """
    fuzzy_relations = defaultdict(dict)
    vocab_size = len(word2idx)

    print(f"{TerminalColors.OKBLUE}構建模糊關係映射中，詞彙總數：{vocab_size}{TerminalColors.ENDC}")
    for i in tqdm(range(vocab_size), desc=f"{TerminalColors.OKCYAN}構建模糊關係映射{TerminalColors.ENDC}", leave=True):
        word_i_index = i
        related_words = {}
        for j in range(vocab_size):
            if i == j:
                continue
            similarity = similarity_matrix[i, j]
            if similarity > similarity_threshold:
                word_j_index = j
                related_words[word_j_index] = float(similarity)  # 隸屬度設定為相似度
        if related_words:
            fuzzy_relations[word_i_index] = related_words

    print(f"{TerminalColors.OKGREEN}具有模糊關係的詞彙數量：{len(fuzzy_relations)}{TerminalColors.ENDC}")
    return fuzzy_relations

def main():
    # 設定資料路徑
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data', 'raw', 'phoenix14t', 'PHOENIX-2014-T.train.corpus.csv')
    embedding_model_path = os.path.join(project_root, 'data', 'embeddings', 'cc.de.300.bin')

    # 加入日期到輸出檔案名稱
    date_str = datetime.now().strftime('%Y%m%d')
    output_path = os.path.join(project_root, 'data', 'processed', f'fuzzy_relations_{date_str}.pkl')

    # 檢查檔案是否存在
    print(f"{TerminalColors.HEADER}檢查資料檔案：{data_path}{TerminalColors.ENDC}")
    if not os.path.exists(data_path):
        print(f"{TerminalColors.FAIL}資料檔案未找到：{data_path}{TerminalColors.ENDC}")
        return
    print(f"{TerminalColors.HEADER}檢查詞嵌入模型檔案：{embedding_model_path}{TerminalColors.ENDC}")
    if not os.path.exists(embedding_model_path):
        print(f"{TerminalColors.FAIL}詞嵌入模型未找到：{embedding_model_path}{TerminalColors.ENDC}")
        return
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    # 載入資料並構建詞彙表
    print(f"{TerminalColors.OKCYAN}開始載入資料並構建詞彙表...{TerminalColors.ENDC}")
    german_sentences = load_phoenix14t_data(data_path)
    german_vocab, word2idx, idx2word = build_vocab(german_sentences)
    print(f"{TerminalColors.OKGREEN}詞彙表構建完成，詞彙數量：{len(german_vocab)}{TerminalColors.ENDC}")

    # 載入詞嵌入模型
    print(f"{TerminalColors.OKCYAN}載入預訓練的德語詞嵌入模型...{TerminalColors.ENDC}")
    embedding_model = load_embeddings(embedding_model_path)
    print(f"{TerminalColors.OKGREEN}詞嵌入模型載入完成{TerminalColors.ENDC}")

    # 計算相似度矩陣
    print(f"{TerminalColors.OKCYAN}計算詞彙之間的相似度...{TerminalColors.ENDC}")
    similarity_matrix, oov_words = compute_similarity_matrix(german_vocab, embedding_model)

    print(f"{TerminalColors.WARNING}OOV（未登錄）詞彙數量：{len(oov_words)}{TerminalColors.ENDC}")

    # 構建模糊關係映射
    print(f"{TerminalColors.OKCYAN}開始構建模糊關係映射...{TerminalColors.ENDC}")
    fuzzy_relations = build_fuzzy_relations(similarity_matrix, word2idx, idx2word, similarity_threshold=0.7)

    # 保存詞彙表和模糊關係映射，並添加進度條
    vocab_data = {'word2idx': word2idx, 'idx2word': idx2word}
    vocab_path = os.path.join(project_root, 'data', 'processed', f'vocab_{date_str}.pkl')
    print(f"{TerminalColors.OKBLUE}保存詞彙映射到 {vocab_path}...{TerminalColors.ENDC}")
    with tqdm(total=1, desc=f"{TerminalColors.OKCYAN}保存詞彙映射{TerminalColors.ENDC}") as pbar:
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab_data, f)
        pbar.update(1)
    print(f"{TerminalColors.OKGREEN}詞彙映射已保存到 {vocab_path}{TerminalColors.ENDC}")

    # 保存模糊關係映射
    print(f"{TerminalColors.OKBLUE}保存模糊關係映射到 {output_path}...{TerminalColors.ENDC}")
    with tqdm(total=1, desc=f"{TerminalColors.OKCYAN}保存模糊關係映射{TerminalColors.ENDC}") as pbar:
        with open(output_path, 'wb') as f:
            pickle.dump(fuzzy_relations, f)
        pbar.update(1)
    print(f"{TerminalColors.OKGREEN}模糊關係映射已保存到 {output_path}{TerminalColors.ENDC}")

if __name__ == '__main__':
    main()

