import pickle
import os
import networkx as nx
import matplotlib.pyplot as plt

# 設定專案根目錄
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 設定檔案路徑
fuzzy_relations_path = os.path.join(project_root, 'data', 'processed', 'fuzzy_relations.pkl')
vocab_path = os.path.join(project_root, 'data', 'processed', 'vocab.pkl')

# 加載模糊關係映射
with open(fuzzy_relations_path, 'rb') as f:
    fuzzy_relations = pickle.load(f)

# 加載詞彙映射
with open(vocab_path, 'rb') as f:
    vocab_data = pickle.load(f)
    word2idx = vocab_data['word2idx']
    idx2word = vocab_data['idx2word']

# 構建 NetworkX 圖形
G = nx.Graph()

# 遍歷模糊關係映射，將詞彙及其關係加入圖中
for target_idx, related_words in list(fuzzy_relations.items())[:10]:  # 只展示前10個詞彙
    target_word = idx2word[target_idx]
    G.add_node(target_word)  # 添加節點
    for related_idx, degree in related_words.items():
        related_word = idx2word[related_idx]
        G.add_edge(target_word, related_word, weight=degree)  # 添加邊，權重為隸屬度

# 設定節點和邊的外觀
pos = nx.spring_layout(G)  # 使用 spring 布局進行圖形排列
weights = [G[u][v]['weight'] for u, v in G.edges()]  # 取得邊的權重

# 創建子圖來管理 colorbar 和圖形的佈局
fig, ax = plt.subplots()

# 繪製邊並根據權重設置顏色和寬度
edges = nx.draw_networkx_edges(G, pos, ax=ax, edge_color=weights, width=[w * 2 for w in weights], edge_cmap=plt.cm.Blues)

# 繪製節點與標籤
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=500, node_color='lightblue')
nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_family='sans-serif')

# 加入邊的顏色條（colorbar）表示隸屬度
sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min(weights), vmax=max(weights)))
sm.set_array(weights)
plt.colorbar(sm, ax=ax, label='隸屬度')  # 指定 colorbar 應該繪製在哪個軸上

# 儲存並顯示圖形
output_image_path = os.path.join(project_root, 'data', 'processed', 'fuzzy_relations_graph.png')
plt.title("模糊關係圖")
plt.savefig(output_image_path)
plt.show()

print(f"模糊關係圖已保存為 {output_image_path}")

