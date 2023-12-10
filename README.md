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
