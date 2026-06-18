# Task1：PWWS 文本对抗样本攻击实验报告

## 1. 论文介绍

本任务复现论文 *Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency* 中的 PWWS 方法。该论文关注文本分类任务中的对抗样本生成问题。与图像攻击不同，文本由离散 token 组成，不能直接在连续像素空间中加入微小扰动；同时，文本扰动必须尽量保持词法正确、语法正确和语义相似，否则生成的样本很容易被人发现，也不再是有效的自然语言对抗样本。

PWWS 的威胁模型属于黑盒模型：攻击者不需要访问模型结构和梯度，只需要查询分类器得到每个类别的预测概率。攻击者的操作空间是对输入句子中的词进行同义词替换。算法先用 WordNet 为每个可替换词生成候选同义词，然后评估每个替换对真实标签概率的降低程度；同时，通过 leave-one-out 的方式计算词重要性，即把某个词替换为 `[UNK]` 后观察模型真实标签概率变化。最后，PWWS 将词重要性和替换攻击效果相乘，得到最终排序，并按该顺序贪心替换词，直到模型预测标签改变或搜索结束。

PWWS 的优点是实现相对直接，不依赖模型梯度，适用于只有预测概率接口的分类器；同时，它通过同义词替换和较低修改率尽量保持文本可读性。缺点是 WordNet 同义词不一定适合当前上下文，可能产生语义偏移或不自然替换；此外，算法需要大量模型查询，攻击效率受候选词数量影响明显。

## 2. PWWS 算法描述

给定原始文本 $x = w_1, w_2, ..., w_n$ 和真实标签 $y$，PWWS 的目标是在尽量少修改词的情况下构造 $x_{adv}$，使得：

$$
f(x_{adv}) \neq y
$$

算法分为三个步骤。

第一，生成候选同义词。对每个可修改词 $w_i$，使用 WordNet 得到候选集合 $L_i$，过滤原词、多词短语和非英文词。

第二，计算词重要性。对每个词做 leave-one-out 操作，将其替换为 `[UNK]`，观察真实标签概率变化：

$$
S_i = P(y|x) - P(y|x_{\backslash i})
$$

如果去掉某个词后真实标签概率下降明显，说明该词对原分类结果更重要。

第三，计算替换效果并排序。对每个词的候选同义词，选择使真实标签概率下降最多的替换。再将该替换效果与词重要性经过 softmax 后的权重相乘，得到 PWWS 分数。算法按照 PWWS 分数从高到低贪心替换，直到预测标签改变。

伪代码如下：

```text
Input: text x, true label y, classifier F
for each modifiable word wi:
    build synonym set Li using WordNet
    compute leave-one-out saliency Si
    for each synonym s in Li:
        replace wi with s and query F
        compute attack score = 1 - P(y | replaced text)
    keep best synonym score for wi
rank words by softmax(S) * best synonym score
current_text = x
for word in ranked order:
    try its synonyms in current_text
    accept the synonym that most reduces P(y)
    if prediction label changes:
        return adversarial example
return failed example
```

## 3. 代码补全说明

本实验补全 `nlp_adv_2026.py` 中四个核心 TODO。

### 3.1 `get_wordnet_synonyms`

该函数使用 NLTK WordNet 生成同义词候选。实现时遍历 synset 和 lemma，将候选词统一转为小写，并过滤三类不合适候选：与原词相同的词、包含下划线的多词短语、非纯字母英文词。这样可以保证候选词基本符合“单词级替换”的要求。

核心逻辑为：

```python
for synset in wordnet.synsets(word, lang=language):
    for lemma in synset.lemma_names(lang=language):
        candidate = lemma.lower()
        if "_" in candidate: continue
        if not candidate.isalpha(): continue
        if candidate == original: continue
        synonyms.add(candidate)
```

### 3.2 `attack_score`

本实验是非目标攻击，目标是降低真实标签概率。因此攻击分数定义为：

```python
score = 1.0 - probs[true_label]
```

分数越大，说明模型对真实标签越不自信，攻击效果越强。

### 3.3 `rank_words`

`rank_words` 实现 PWWS 的词排序。首先筛选可修改词，再对每个词构造 leave-one-out 文本，计算 saliency；然后对每个词的候选同义词分别查询模型，得到最佳替换攻击分数。最后使用：

```text
PWWS score = softmax(saliency) * best synonym attack score
```

对所有词排序，得到后续贪心攻击顺序。

### 3.4 `attack_one`

`attack_one` 实现单条样本的完整攻击。算法先查询干净样本，如果模型原本就分类错误，则记为 skipped；否则按 PWWS 排序依次尝试替换词。每一步选择当前能最大化攻击分数的同义词替换，如果替换后模型预测标签改变，则攻击成功并返回结果；如果所有可替换词都尝试完仍未改变预测，则攻击失败。

## 4. 实验设置与 Baseline 结果

实验数据集为 SST-2，模型为 `distilbert/distilbert-base-uncased-finetuned-sst-2-english`。运行 30 条 dev 样本，并启用 shuffle。运行命令如下：

```bash
python nlp_adv_2026.py \
  --sst2-zip SST-2.zip \
  --split dev \
  --limit 30 \
  --shuffle \
  --download-wordnet \
  --model-name-or-path distilbert/distilbert-base-uncased-finetuned-sst-2-english \
  > task1_pwws_limit30.log
```

Baseline 总体结果如下：

| 指标 | 结果 |
|---|---:|
| total | 30 |
| clean_correct | 27 |
| successful | 25 |
| failed | 2 |
| skipped | 3 |
| attack_success_rate | 0.9259 |
| avg_queries | 93.11 |
| avg_changed_words | 2.19 |

可以看到，在 27 条原本分类正确的样本中，PWWS 成功攻击 25 条，攻击成功率为 **92.59%**。平均查询次数为 **93.11**，平均修改词数为 **2.19**。这说明 PWWS 在 SST-2 情感分类任务上攻击能力较强，通常只需要替换少量词就能改变模型预测。

## 5. 攻击样本分析

### 5.1 成功且较自然的样本

原句：

```text
the heavy-handed film is almost laughable as a consequence .
```

对抗样本：

```text
the heavy - handed film is almost comic as a consequence.
```

替换词：`laughable -> comic`。模型标签从 0 变为 1，查询次数为 64。这个替换在语义上比较自然，`laughable` 与 `comic` 都与“可笑”相关，句子可读性较好，说明 PWWS 在部分样本上能够生成较自然的对抗文本。

### 5.2 成功但语义质量较差的样本

原句：

```text
it 's not that kung pow is n't funny some of the time -- it just is n't any funnier than bad martial arts movies are all by themselves , without all oedekerk 's impish augmentation .
```

对抗样本中出现：

```text
n -> normality, t -> thymine
```

模型标签从 0 变为 1，攻击成功，但替换明显不自然。这类现象来自分词后的短 token，被 WordNet 映射到罕见词或专有含义，破坏了句子可读性。

### 5.3 失败样本

原句：

```text
they should have called it gutterball .
```

对抗样本：

```text
they should have foretell it gutterball.
```

替换词：`called -> foretell`。模型标签仍为 0，攻击失败，查询次数为 23。该例说明，即使替换词能够改变局部语义，也不一定足以跨越模型分类边界；同时该替换语法上也不自然，说明 WordNet 候选需要进一步约束上下文适配性。

## 6. 算法优化与进一步实验

原始 PWWS 的主要问题是：WordNet 候选只保证词典层面的相关性，不保证在当前句子中自然；同时短 token、罕见词和长度差异过大的候选可能导致明显语义偏移。因此，我实现了优化版 `nlp_adv_2026_opt.py`，并将结果保存在 `task1_results_opt/`，避免覆盖 baseline 结果。

优化主要包括：

1. 设置 `min_word_len=3`，避免替换 `n`、`t` 等短 token。
2. 使用 WordNet lemma count 过滤过于罕见的候选词。
3. 限制候选词与原词长度比例，减少形式差异过大的替换。
4. 使用 `max_change_ratio` 限制最大修改比例，提高语义保持性。
5. 调整 `max_candidates`，观察攻击效率和成功率之间的权衡。

优化实验结果如下：

| 方法 | 参数设置 | ASR | 平均查询次数 | 平均修改词数 | 主要观察 |
|---|---|---:|---:|---:|---|
| baseline | `max_candidates=10` | 0.9259 | 93.11 | 2.19 | 攻击成功率最高，但有明显不自然替换 |
| opt_semantic | `min_len=3, ratio=0.2, max_candidates=10` | 0.5185 | 56.63 | 1.74 | 修改更少，短词问题减少，但成功率明显下降 |
| opt_fast | `min_len=3, ratio=0.2, max_candidates=5` | 0.4815 | 44.30 | 1.74 | 查询次数最低，但攻击能力进一步下降 |
| opt_success | `min_len=3, ratio=0.4, max_candidates=20` | 0.7037 | 67.63 | 2.37 | 成功率高于 opt_semantic，但语义自然性仍不稳定 |

从结果看，baseline 的攻击成功率最高，但代价是候选词质量不稳定，例如 `exciting -> sex`、`pure -> gross`、`arms -> munition` 等替换会明显改变语义。`opt_semantic` 限制修改比例和短词替换后，平均查询次数从 93.11 降到 56.63，平均修改词数从 2.19 降到 1.74，但 ASR 降至 0.5185。说明更严格的语义约束会减少可用搜索空间，从而牺牲攻击成功率。

`opt_fast` 进一步减少候选词数量，平均查询次数降到 44.30，验证了减少候选数量可以提升攻击效率，但 ASR 也降至 0.4815。`opt_success` 增加候选数量并放宽最大修改比例后，ASR 回升到 0.7037，但平均修改词数上升到 2.37，说明攻击成功率、查询效率和语义保持性之间存在明显权衡。

## 7. 总结

本实验完成了 PWWS 文本对抗攻击的核心代码补全，并在 SST-2 情感分类任务上验证了攻击效果。Baseline 在 30 条样本上取得 92.59% 的攻击成功率，说明基于同义词替换和词重要性排序的黑盒攻击可以有效欺骗情感分类模型。

进一步实验表明，提高语义保持性通常会降低攻击成功率；减少候选词数量可以降低查询次数，但会削弱攻击能力；放宽修改比例和增加候选词可以提高成功率，但可能损害文本自然性。因此，文本对抗攻击不能只看 ASR，还需要同时关注查询效率、修改词数和语义保持性。后续可以引入上下文语言模型、词性约束或句向量相似度筛选候选词，使替换更符合上下文语义。
