# Persian Word Embedding
Persian Word Embedding using FastText, BERT, GPT and GloVe

### 1. How to use FastText Embedding 
1.1 How to install fasttext:
```
pip install fasttext
pip install huggingface_hub
```

1.2 Here is how to load and use a pre-trained vectors:
```
import fasttext
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="facebook/fasttext-en-vectors", filename="model.bin")
model = fasttext.load_model(model_path)
model.words

['رسانه', 'بورس', 'اعضای', 'دیده', 'عملکرد', 'ویرایش', 'سفارش', 'کارشناسی', 'کلاه', 'کمتر', ...]

len(model.words)
2000000

model['زندگی']

array([ 4.89417791e-01,  1.60882145e-01, -2.25947708e-01, -2.94273376e-01,
       -1.04577184e-01,  1.17962055e-01,  1.34821936e-01, -2.41778508e-01, ...])
```

1.3 Here is how to use this model to query the **nearest neighbors** of a Persian word vector:

```

model.get_nearest_neighbors("بورس", k=5)

[(0.6276253461837769, 'سهام شاخص'),
 (0.6252498626708984, 'معاملات'),
 (0.6190851330757141, 'بهادار'),
 (0.6184772253036499, 'اقتصادبورس'),
 (0.6100088357925415, 'بورسهر')]
```

