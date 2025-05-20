import jieba
from gensim.models import Word2Vec

# 1. 读取停用词
with open('stop_words.txt', 'r', encoding='utf-8') as f:
    stopwords = set(line.strip() for line in f if line.strip())

# 2. 读取歌词语料
with open('lyrics_10k.txt', 'r', encoding='utf-8') as f:
    lyrics_corpus = [line.strip() for line in f if line.strip()]

# 3. 分词并去除停用词
tokenized_corpus = []
for song in lyrics_corpus:
    words = list(jieba.cut(song))
    filtered = [w for w in words if w not in stopwords]
    tokenized_corpus.append(filtered)

# 4. 初始化并建立词表
model = Word2Vec(vector_size=100, window=5, min_count=1, workers=8)
model.build_vocab(tokenized_corpus)

# 5. 多轮训练
model.train(tokenized_corpus, total_examples=model.corpus_count, epochs=200)

# 6. 保存模型
model.save("lyrics_word2vec.model")

# 7. 输出示例词的同义词
sample_words = ['我', '爱', '歌曲', '音乐', '生活']
for word in sample_words:
    try:
        sims = model.wv.most_similar(word, topn=3)
        print(f"同义词——{word}: {sims}")
    except KeyError:
        print(f"‘{word}’ 不在词表中")