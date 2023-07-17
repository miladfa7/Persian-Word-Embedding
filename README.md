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
-----------------------

### 2. How to use BERT(ParsBERT) Embedding 
2.1 How to install huggingface:
```
pip install transformers
```
2.2 Here is how to load and use a pre-trained vectors:

```
from transformers import BertTokenizer, BertModel

model_name = 'HooshvareLab/bert-fa-zwnj-base'  # Specify the BERT model variant
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

text = "جز عشق نبود هیچ دم ساز مرا نی اول "
tokenizer.tokenize(text)

['جز', 'عشق', 'نبود', 'هیچ', 'دم', 'ساز', 'مرا', 'نی', 'اول']

```
2.3 Here how to get **word embedding** of bert:
```
encoded_input = tokenizer.encode_plus(
    text,
    add_special_tokens=True,
    padding='max_length',
    truncation=True,
    max_length=150,  # Specify the desired maximum length of the sequence
    return_tensors='pt' 
  )
input_ids = encoded_input['input_ids']
attention_mask = encoded_input['attention_mask']
with torch.no_grad():
  outputs = model(input_ids, attention_mask=attention_mask)
  words_embedding = outputs.last_hidden_state

words_embedding

tensor([[[ 0.2545, -0.3399,  0.0990,  ...,  0.3291,  0.3309,  1.2594],
         [ 0.5799, -0.1835, -0.1979,  ...,  0.7980, -0.3029, -0.1636],
         [ 0.4741, -0.1815, -0.0451,  ...,  1.8211,  0.1717, -0.3972],
         ...,
         [-0.3178, -0.9737,  0.5525,  ...,  0.4877,  0.1396,  0.7577],
         [ 0.1801, -0.8703,  0.2300,  ...,  0.4041,  0.4268,  0.5552],
         [-0.4429, -0.3841,  0.8476,  ...,  0.3903,  0.8899,  1.8148]]])

words_embedding.size()
torch.Size([1, 150, 768]) # batch_size, max_len, embedding_dim

```
2.4 Here how to get **sentence embedding** of bert:
```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sentence_embedding = torch.mean(embeddings, dim=1).to(device)  # Shape: [1, 768]
sentence_embedding = sentence_embedding.squeeze(0)
sentence_embedding

tensor([ 8.5537e-02, -7.5624e-01,  1.9884e-01, -7.9048e-01, -1.6724e+00,
        -1.0927e+00, -3.7952e-01, -5.0552e-01, -6.3537e-01,  1.5239e+00,
        -8.8235e-01,  4.4737e-01, -5.0677e-01, -9.2339e-01, -8.2049e-01,
         3.1416e-03, -1.5347e-01, -5.0761e-01, -1.2381e+00,  1.3580e-01,
       ...
])
```





