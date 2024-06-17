url = "https://hf-mirror.com/2Noise/ChatTTS/tree/main"
from pycrawlers import huggingface

# 实例化类
hg = huggingface()

urls = [url]

# 批量下载
# 默认保存位置在当前脚本所在文件夹 ./
# hg.get_batch_data(urls)

# 自定义下载位置
paths = ['/Users/zhangsan/workspace/model_hg_temp']
hg.get_batch_data(urls, paths)
