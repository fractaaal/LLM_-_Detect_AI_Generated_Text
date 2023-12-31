## はじめに

kaggle の勉強として初めて Competitions に参加しました。
勉強としての参加なので、Notebook をこのレポジトリで管理していこうと思います。

## Competitions 概要

「LLM - Detect AI Generated Text」というコンペに参加しました。
概要は[こちら](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/overview)

### 2023/12/10

初めて挑戦するコンペなので Fork しないで初めから書いてみた。
以下を実施。
notebook: `notebook/competitions/LLM_-_Detect_AI_Generated_Text/nb001/first.ipynb`

- TF-IDF でトークン化、ロジスティック回帰でパイプラインを組む
- K 分割交差検証（5 分割）で各 AUC を算出

CV の結果は以下。

```
Fold 1: AUC = 0.9993
Fold 2: AUC = 0.9989
Fold 3: AUC = 0.9991
Fold 4: AUC = 0.9992
Fold 5: AUC = 0.9992

Mean AUC = 0.9991
```

submit してみたが結果は score=0.823 で、LeaderBoard 1595/2292、、、
CV との乖離が大きいので過学習の恐れがある。
モデルのチューニングや特徴量エンジニアリングをもう少し勉強する必要がある。

### 2023/12/11

特徴量選択で以下をパイプラインに追加するも LB の AUC=0.723 となり、不採用
`selector = SelectKBest(score_func=chi2, k=1000) `

### 2023/12/12

tokenizer を追加し、TF-IDF ベクトル化器に引数として渡すようにした。
これも LB の AUC=0.814 で最初のものより低くなってしまった。
`notebook/competitions/LLM_-_Detect_AI_Generated_Text/nb002/add_tokenizer.ipynb`

```.python
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()

# カスタムトークナイザーとステミングを組み合わせてトークン化する
def my_tokenizer(text):
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

def custom_preprocessor(text):
    # テキストを小文字に変換
    text = text.lower()
    text = text.replace(".", "")
    return text

```

tokenizer を修正したもう一度行うと LB：0.824 となった。
`notebook/competitions/LLM_-_Detect_AI_Generated_Text/nb002/add_stemmer.ipynb`

### 2023/12/14

1. アンサンブル学習の利用

アンサンブル学習の VotingClassifier を使用して多数決を行うようにした。
LB の結果は 0.84 となり、現段階でベストスコアとなった。

`notebook/competitions/LLM_-_Detect_AI_Generated_Text/nb003/first.ipynb`

```.python
# TF-IDFベクトル化器を作成
tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

test_x = tfidf.fit_transform(test["text"])
train_x = tfidf.transform(train["text"])

lr = LogisticRegression()
clf = MultinomialNB(alpha=0.02)
sgd_model = SGDClassifier(max_iter=5000, tol=1e-3, loss="modified_huber")
sgd_model2 = SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber", class_weight="balanced")
sgd_model3 = SGDClassifier(max_iter=10000, tol=5e-4, loss="modified_huber", early_stopping=True)

ensemble = VotingClassifier(
    estimators=[
        ("lr", lr),
        ("mnb", clf),
        ("sgd", sgd_model),
        ("sgd2", sgd_model2),
        ("sgd3", sgd_model3),
    ],
    voting="soft",
)
ensemble.fit(train_x, train.label)

```

2. カスタムトークナイザーを追加

アンサンブルのモデルに対し、トークナイザーを下記のように設定すると LB のスコア 0.89 となり、現時点でベストスコアとなった。

`notebook/competitions/LLM_-_Detect_AI_Generated_Text/nb004/add_tokenizer.ipynb`

```.python
# ストップワードのロード
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # HTMLタグの削除
    text = re.sub(r'<.*?>', '', text)
    # 句読点、数字、特殊記号の削除
    text = re.sub(r'[^\w\s]', '', text)
    # 小文字化
    text = text.lower()
    return text

# カスタムトークナイザー
def tokenize(text):
    tokens = word_tokenize(text)
    return [word for word in tokens if word not in stop_words]
```

### 2023/12/15

tf-idf ベクトルの特徴量が大きすぎる可能性があるため、max_features オプションを追加して 500000 に設定してみた。（サイズは(44868, 19803183)）

結果は LB スコア 0.921 となりベストスコア。やはり過学習を克服していく方向性で進めていこうと思う。

### 2023/12/16

特徴量削減の最適解を探索していく。結果は以下。

| max_features | LB score(AUC) |
| :----------- | :-----------: |
| 100000       |     0.903     |
| 300000       |     0.918     |
| 500000       |     0.921     |
| 600000       |     0.921     |
| 750000       |     0.921     |
| 1000000      |     0.914     |

max_features=500000 を設定するのが良さそう。
