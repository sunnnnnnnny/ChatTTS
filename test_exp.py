# simple usage
import os
import torch
import ChatTTS
import soundfile
from IPython.display import Audio
import torchaudio

model_dir = "/Users/zhangsan/workspace/model_hg_temp/ChatTTS"
chat = ChatTTS.Chat()
chat.load_models(compile=False) # Set to True for better performance

texts = ["tomorrow is another day"]
texts = ["很多人觉得，想把英语学的好，单词一个不能少。一个个的是死背单词。知道的单词多了当然会是好事。可是除了考试以外，或是在写作阅读以外，在我们中国式的哑巴英语上，我们缺少是词汇量么?"]

wavs = chat.infer(texts, lang="zh")

# torchaudio.save("output1.wav", torch.from_numpy(wavs[0]), 24000)
outdir = "output"
if not os.path.exists(outdir):
    os.makedirs(outdir)
soundfile.write(os.path.join(outdir, "output1.wav"), wavs[0][0], 24000)