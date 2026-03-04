# nlp2026_spandetect
言語処理学会 第32回年次大会で発表した、「[マスク予測モデルを用いた軽量なハルシネーションのスパン検出手法](https://anlp.jp/proceedings/annual_meeting/2026/pdf_dir/B1-19.pdf)」の実装

<img src="method_3.png" width="600" alt="ロゴ">

## 環境について
3つのrequirementsファイルがある。
- `requirements_bert.txt` : ModernBERTを用いているコードを実行するための環境
    - Python 3.11.8
- `requirements_srl.txt` : AllenNLPのSRLモデルを用いているコードを実行するための環境
    - Python 3.10.14
- `requirements_llama.txt` : Llamaを用いているコードを実行するための環境
    - Python 3.12.11
## data/
実験には[RAGTruthデータセット](https://github.com/ParticleMedia/RAGTruth) (Wu+, 2023) から、QAタスク、要約タスクのものを用いた。
### source_info.jsonl
RAGTruthデータセットの入力文章。(QAタスクでは回答の参考になる文章、要約タスクでは要約前の文章)
### response.jsonl
RAGTruthデータセットの出力文章。(QAタスクでは回答、要約タスクでは要約後の文章)
### ft_{train, dev, test}.jsonl
比較手法を実験するためのデータセット。
主要なフィールドは以下の通り。
| Field name| Field value | Description |
| --- | --- | --- |
| id_name | string | データセット内の一意なID |
|source_id| string | source_info.jsonlのsource_idと対応 |
|task_type| string | "QA" or "Summary" |
|model| string | 出力文章を生成したモデル名(6種類) |
|source_info|dict or string|入力文章。QAタスクの場合は辞書型で、"question"と"passage"をキーに持つ。要約タスクの場合は文字列型で、要約前の文章。|
|response|string|出力文章|
|labels|list[dict]|出力文章中のハルシネーションのスパンのリスト。各スパンは、"start" (スパンの開始文字位置)、"end" (スパンの終了位置)、"text" (スパンのテキスト)、"label_type" (ハルシネーションのタイプ(4種類)) などをキーに持つ辞書型。|
|input_text|string|Llamaにハルシネーションのスパンを予測させるためのプロンプト (RAGTruthに準拠) 。source_infoとresponseを組み合わせて作成されている。|

### 1127_srl_{train, cls, dev, test_hal}.jsonl
提案手法を実験するためのデータセット。
- train: ModernBERTのfine-tuning用のデータセット
- cls: 線形分類器の学習用のデータセット
- dev: 開発データセット
- test_hal: 評価データセット (ハルシネーションを全く含まない文章は除外)
先述のft_{train, dev, test}.jsonlと同様のフィールドに加えて、以下のフィールドがある。
| Field name| Field value | Description |
| --- | --- | --- |
| srl_splits | list[dict] | 出力文章をSRLで分割したスパンのリスト。各スパンは、"start" (スパンの開始文字位置)、"end" (スパンの終了位置)、"text" (スパンのテキスト) 、"sentence_index" (スパンが属する文のインデックス)、"token_span" (スパンが属するトークンの開始位置と終了位置) などをキーに持つ辞書型。|
|sentence_ids| list[int] | 入力文章の各トークンが何番目の文に属するかを示すリスト。|
## create_ft_dataset.py
`source_info.jsonl`と`response.jsonl`を組み合わせて、`ft_{train, dev, test}.jsonl`を作成するスクリプト。
## create_srl_dataset.py
`ft_{train, dev, test}.jsonl`をもとに、SRLでスパンを分割して、`1127_srl_{train, cls, dev, test_hal}.jsonl`を作成するスクリプト。
実行引数として"--mode"をとり、"train"、"dev"、"test"のいずれかを指定する。
SRLモデルにはAllenNLPの[SRLモデル](https://docs.allennlp.org/models/main/models/structured_prediction/predictors/srl/)を使用している。

スパン分割のルールは以下の通り。
- 動詞の数だけ分割結果があるが、一旦全ての分割結果を採用し、最も細かい分割結果を作成する。
    - 原文 : Thomas was charged with attempting to travel to Syria to join ISIS.
    - 分割結果1 : [ARG1: Thomas] was [V: charged] [ARG2: with attempting to travel to Syria to join ISIS] .
    - 分割結果2 : [ARG0: Thomas] was charged with [V: attempting] [ARG1: to travel to Syria to join ISIS] .
    - 分割結果3 : [ARG0: Thomas] was charged with attempting to [V: travel] [ARG1: to Syria] [ARGM-PRP: to join ISIS] .
    - 分割結果4 : [ARG0: Thomas] was charged with attempting to travel to Syria to [V: join] [ARG1: ISIS] .
    - 最も細かい分割結果 : Thomas / was / charged / with / attempting / to / travel / to Syria / to / join / ISIS.
- 受動態や進行形など、複数の動詞が連続している場合や、前置詞や不定詞が独立してしまっている場合は、分割結果を統合する。
    - 最終的な分割結果 : Thomas / was charged / with attempting / to travel / to Syria / to join / ISIS.

## modernbert_baseline.py
ModernBERTを用いたトークン分類タスクのfine-tuning手法の実装。
入力文章と出力文章を結合して入力し、出力文章中の各トークンがハルシネーションに含まれるかどうかを予測する。
## modernbert_inference.py
`modernbert_baseline.py`でfine-tuningしたモデルを用いて、テストデータに対して推論を行う。
性能評価は文字単位で行われる。
## llama_ft.py
プロンプトを用いて、Llamaにハルシネーションのスパンを予測させるタスクのfine-tuning手法の実装。
入力文章と出力文章を含んだプロンプトを入力して、出力文章中のハルシネーションのスパンをjson形式で出力させる。

実行コマンド例
```
python llama_ft.py --model_name meta-llama/Llama-3.1-8B-Instruct --out_dir ft_results/llama_ft 
```
## llama_inference.py
`llama_ft.py`でfine-tuningしたモデルを用いて、テストデータに対して推論を行う。

実行コマンド例
```
python llama_inference.py --base_model meta-llama/Llama-3.1-8B-Instruct --lora_dir ft_results/llama_ft --out ft_results/llama_ft.jsonl
```
## llama_eval.py
`llama_inference.py`で得られた予測結果をもとに、性能評価を行うスクリプト。
Llamaの出力はjson形式になっているため、その中からスパンテキストを抽出し、出力文章のどの位置なのかを特定する処理も含まれている。
性能評価は文字単位で行われる。
## utils/
提案手法の実装に用いるクラスや関数がまとまっている。
### judge_include.py
提案手法では、出力文章中の各チャンクをマスクし、マスク箇所を入力文章中から抜き出して予測する操作を通してハルシネーションのスパンを検出する。
そのため、訓練データの出力文章の各チャンクに対して、マスク箇所を埋めることのできる入力文章中のスパン集合 (Y+) を定義する必要がある。
このスクリプトでは、マスク箇所を埋めることのできる入力文章中のスパンを定義する関数を実装している。
- include_ngram
    - 出力文章中のマスクされたチャンクとn-gramが類似している入力文章中のスパンをY+とする。
    - 「類似」しているとは、2/3以上のn-gramが共通していることと定義する。ただし、"the"や"to"などの汎用的な単語のみのスパンは除外する。
- include_ngram_perfect
    - 出力文章中のマスクされたチャンクと完全に一致している入力文章中のスパンをY+とする。
### mask_data.py
出力文章中のチャンクをマスクするための関数が実装されている。
訓練時はY+が空集合であるチャンクはマスクされない。
推論時は全てのチャンクがマスクされる。
- mask_text
    - 出力文章中の1つの与えられたチャンクを2つのマスクトークンで置き換える関数。
- mask_text_multiple
    - 出力文章中の複数の与えられたチャンクをそれぞれ2つのマスクトークンで置き換える関数。
- mask_data
    - 1つの出力文章につき、1つのチャンクをランダムに選択して、`mask_text`を用いてマスクする関数。
    - ハルシネーションが含まれる出力文章の場合は、ハルシネーションのチャンクを優先的にマスクする。
- mask_data_some
    - 1つの出力文章につき、複数のチャンクをランダムに選択して、`mask_text`を用いてマスクする関数。
    - 1つの出力文章につき、複数のチャンクがマスクされ、それぞれ違う事例として扱う。
    - マスクするチャンクは幾何分布に従ってランダムに選択される。
    - ハルシネーションが含まれる出力文章の場合は、ハルシネーションのチャンクを優先的にマスクする。
- mask_data_multi
    - 1つの出力文章につき、複数のチャンクをランダムに選択して、`mask_text_multi`を用いてマスクする関数。
    - 1つの出力文章につき、複数のチャンクがマスクされ、1つの事例として扱う。
    - マスクするチャンクは幾何分布に従ってランダムに選択される。
    - ハルシネーションが含まれる出力文章の場合は、ハルシネーションのチャンクを優先的にマスクする。
### data.py
Trainerクラスに対応したデータ形式に変換するための関数やクラスが実装されている。
### loss.py
提案手法をfine-tuningする際の損失関数や、推論時に付与する予測スコアを計算する関数が実装されている。
損失関数は、マスクされたチャンクがハルシネーションを含まない場合はY+のチャンクが予測されるように、マスクされたチャンクがハルシネーションを含む場合はY+のチャンクが予測されないように、モデルを訓練するためのものである。
contrastive_lossとmargin_lossの2種類が実装されている。
### model.py
提案手法のモデルが実装されている。
訓練時と推論時両方に共通する処理が`HalNPMBase`クラスに実装されている。
Fine-tuningの際の処理が`HalNPMTrainModel`クラスに実装されている。
推論の際の処理が`HalNPMInferenceModel`クラスに実装されている。
## train_npm.py
提案手法のfine-tuningを行うスクリプト。

実行コマンド例
```
python train_npm.py --mask_mode some --save_name modernbert_margin --loss_mode margin
```
## inference_npm_ft_batch.py
提案手法のfine-tuningしたモデルを用いて、テストデータに対して推論を行うスクリプト。
マスク箇所の予測を行う際は、入力文章中の上位5つの予測スコアを保存しておく。
また、その5つのチャンクに対して、マスクされた元のチャンクとのコサイン類似度も計算して保存しておく。
推論時間は増加するが、予測されたチャンクの文字列を保存することもできる。

実行コマンド例
```
python inference_npm_ft_batch.py --mode test --model_name modernbert_margin --predword 
```
## eval_classifier.ipynb
推論結果をもとに、各チャンクがハルシネーションを含むかどうかを予測する線形分類器を学習し、性能評価を行う。

