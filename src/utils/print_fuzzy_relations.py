import os
import pickle

def print_fuzzy_relations_info():
    # 獲取專案根目錄（根據具體情況修改）
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    # 設定模糊關係映射的路徑
    fuzzy_relations_path = os.path.join(project_root, 'data', 'processed', 'fuzzy_relations.pkl')

    # 確認檔案是否存在
    if not os.path.exists(fuzzy_relations_path):
        print(f"檔案 {fuzzy_relations_path} 不存在")
        return

    # 加載並列印模糊關係映射
    with open(fuzzy_relations_path, 'rb') as f:
        fuzzy_relations = pickle.load(f)
    
    # 列印 fuzzy_relations 資訊
    print("模糊關係資訊:")
    for key, value in fuzzy_relations.items():
        print(f"關鍵詞: {key}")
        print(f"對應值: {value}\n")

if __name__ == "__main__":
    print_fuzzy_relations_info()
