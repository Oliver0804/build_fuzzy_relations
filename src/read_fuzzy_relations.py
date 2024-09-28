import pickle
import os
import networkx as nx
import matplotlib.pyplot as plt

# 設定終端顏色代碼
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

# 設定專案根目錄
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 設定檔案路徑
fuzzy_relations_path = os.path.join(project_root, 'data', 'processed', 'fuzzy_relations.pkl')
vocab_path = os.path.join(project_root, 'data', 'processed', 'vocab.pkl')

# 加載模糊關係映射
print(f"{TerminalColors.OKCYAN}正在載入模糊關係映射檔案：{fuzzy_relations_path}{TerminalColors.ENDC}")
with open(fuzzy_relations_path, 'rb') as f:
    fuzzy_relations = pickle.load(f)
print(f"{TerminalColors.OKGREEN}模糊關係映射已成功載入！{TerminalColors.ENDC}")

# 加載詞彙映射
print(f"{TerminalColors.OKCYAN}正在載入詞彙映射檔案：{vocab_path}{TerminalColors.ENDC}")
with open(vocab_path, 'rb') as f:
    vocab_data = pickle.load(f)
    word2idx = vocab_data['word2idx']
    idx2word = vocab_data['idx2word']
print(f"{TerminalColors.OKGREEN}詞彙映射已成功載入！{TerminalColors.ENDC}")

# 遍歷模糊關係映射，打印部分詞彙及其模糊關係
print(f"{TerminalColors.HEADER}展示前10個目標詞彙及其相關詞彙及隸屬度：{TerminalColors.ENDC}")
def visualize_degree(degree, scale=20):
    """
    使用條形顯示隸屬度。
    
    :param degree: 隸屬度，介於 0 到 1 之間。
    :param scale: 條形的最大長度，預設為 20。
    :return: 表示隸屬度的條形字串。
    """
    bar_length = int(degree * scale)
    return '█' * bar_length + '-' * (scale - bar_length)

for target_idx, related_words in list(fuzzy_relations.items())[:10]:
    target_word = idx2word[target_idx]
    print(f"{TerminalColors.OKBLUE}目標詞彙：{target_word}{TerminalColors.ENDC}")
    for related_idx, degree in related_words.items():
        related_word = idx2word[related_idx]
        degree_bar = visualize_degree(degree)
        print(f"  {TerminalColors.OKCYAN}相關詞彙：{related_word}，隸屬度：{degree:.2f} {degree_bar}{TerminalColors.ENDC}")
    print()  # 空行分隔每個詞彙的結果

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
plt.colorbar(sm, ax=ax, label='Membership Degree')  # 指定 colorbar 應該繪製在哪個軸上

# 儲存並顯示圖形
output_image_path = os.path.join(project_root, 'data', 'processed', 'fuzzy_relations_graph.png')
plt.title("Fuzzy Relations Graph")
plt.savefig(output_image_path)
plt.show()

print(f"{TerminalColors.OKGREEN}模糊關係圖已保存為 {output_image_path}{TerminalColors.ENDC}")

