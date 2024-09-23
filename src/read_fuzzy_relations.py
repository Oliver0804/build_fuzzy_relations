import pickle
import os

# 设置项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 设置文件路径
fuzzy_relations_path = os.path.join(project_root, 'data', 'processed', 'fuzzy_relations.pkl')
vocab_path = os.path.join(project_root, 'data', 'processed', 'vocab.pkl')

# 加载模糊关系映射
with open(fuzzy_relations_path, 'rb') as f:
    fuzzy_relations = pickle.load(f)

# 加载词汇映射
with open(vocab_path, 'rb') as f:
    vocab_data = pickle.load(f)
    word2idx = vocab_data['word2idx']
    idx2word = vocab_data['idx2word']

# 遍历模糊关系映射，打印部分词汇及其模糊关系
for target_idx, related_words in list(fuzzy_relations.items())[:10]:
    target_word = idx2word[target_idx]
    print(f"目标词汇：{target_word}")
    for related_idx, degree in related_words.items():
        related_word = idx2word[related_idx]
        print(f"  相关词汇：{related_word}，隶属度：{degree:.2f}")
    print()
