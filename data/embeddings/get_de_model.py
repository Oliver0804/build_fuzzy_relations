import fasttext.util
import os

# 下載德語模型
fasttext.util.download_model('de', if_exists='ignore')

# 刪除壓縮文件
os.remove('cc.de.300.bin.gz')
