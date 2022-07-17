---
html_meta:
  "description lang=en": "Deep Learning for Molecules & Materials Book"
  "property=og:locale": "en_US"
  "twitter:card": "summary"
  "twitter:description": "Deep Learning for Molecules & Materials Book"
  "twitter:title": "dmol.pub 📖"
  "twitter:image": "https://dmol.pub/_static/robot-chem.png"
  "twitter:site": "@andrewwhite01"
---
![Picture of art installation of networked cables](_static/images/header.png)

# この本の概要

化学や材料科学の分野において、ディープラーニングは標準的なツールになりつつあります。ディープラーニングとは、ある入力データ（特徴量）と出力データ（ラベル）をニューラルネットワークで記述される関数で結びつけることです。ニューラルネットワークは（数値的に）微分可能であり、あらゆる関数を近似することができる強力な道具です。例えば、分子の構造と機能を結びつけることはニューラルネットワークの古典的な活用の一つです。[最近の例](https://doi.org/10.1039/C6SC05720A)では、量子計算を劇的に高速化し、ニューラルネットワークでDFT計算レベルの高い精度を実現できるほどになっています。ディープラーニングが特に注目されるのは、これまで難解だった関数を近似するための強力なツールであること、 **それに加えて、** 新しいデータを生成する能力をもつためです。

この本では、ディープラーニングを、これまで古典的な機械学習では実現不可能だったモデルを構成できるツールの集合体として捉えていきます。ディープラーニングと古典的な機械学習の大きな違いは、特徴量エンジニアリングの必要性です。古典的な機械学習アルゴリズムを用いて予測モデルを構築する際、多くの場合は、データのどのような特徴が重要で、その特徴を分子からどのように計算するかの設計（特徴量エンジニアリング）が必要でした。用いる特徴量は予測性能に大きな影響を与えることから、特徴量エンジニアリングは研究者の頭を悩ませるステップの一つでした。ディープラーニングモデルは、通常、end-to-endで学習されます。すなわち、データのどの特徴が重要であるかという判断はもはや必要無く、分子構造を直接扱うことができるのです。

ディープラーニングがこれほど普及することになったもう一つの理由は、その柔軟性とツールとしての成熟性にあります。以前は、モデルごとに新しい方程式を導出・実装する必要があったため、機械学習におけるモデルの訓練と使用は面倒なものでした。ディープラーニングでは、ニューラルネットワークの学習という一つのアイディアだけで様々な問題に対応できることから、問題ごとに新たな手法を考える必要がなくなり、またモデルの変更も遥かに簡単になりました。ディープラーニングは、科学の新しいパラダイムでもなければ、科学者の代わりでもありません。分子や材料に適用する準備ができた成熟したツールなのです。

## 想定する読者

本書の対象読者は、プログラミングと化学のバックグラウンドを持ち、ディープラーニングを中心としたMaterials Informatics技術の習得に興味のある学生です。例えば、化学や材料科学の大学院生や上級学部生で、ある程度のPythonプログラミングのスキルを持つ人は、本書の恩恵を受けられるでしょう。セクションAとBでは、機械学習の原理を教育的に紹介していますが、機械学習の全てを網羅しているわけではなく、ディープラーニングに必要なトピックのみを取り上げています。例えば、決定木やSVMなどのトピックは、ディープラーニングを理解する上で重要ではないため、カバーされていません。セクションCでは、ディープラーニングの原理と、重要な{doc}`dl/gnn`や{doc}`dl/VAE`などの具体的なアーキテクチャの詳細について説明しています。その他の章では、{doc}`dl/NLP`のように、化学や材料科学を対象とした、より大きなトピックについて概要とサーベイを提供します。最後にセクションDでは、化学と材料科学における実際のディープラーニングの応用例として、より発展的な例を示しています。各セクションの冒頭で必要な背景知識を述べていますが、本書全体を通してPythonプログラミングの基本的なスキルを前提としています。化学に特化したPythonの入門書は、Molecular Sciences Software Instituteの[リソースページ](http://education.molssi.org/resources.html#programming)で見つけることができます。
訳注：日本語で利用できる優れたPythonの入門リソースの一つには、東京大学 数理・情報教育研究センターが公開している[Pythonプログラミング入門 \#utpython](https://utokyo-ipp.github.io/index.html) の講義資料が挙げられます。

## 本書で用いるディープラーニングフレームワークについて
現代のディープラーニングの実装の大半は、ディープラーニングフレームワークの機能の上に成り立っています。したがってフレームワークの選択は学習プロセスの一部といえます。本書では、Pythonと`numpy`に慣れていることは前提とした上で、Pythonによる実装のみを紹介します。ディープラーニングのフレームワークについては、`Jax`、`Tensorflow`、`Keras`を目的に応じて使い分けています。Jaxは基本的にnumpyに自動微分とGPU/TPUアクセレレーションを加えたものなので、習得が簡単です。本書では、実装の詳細を理解し、数式をコードに落とし込むことが重要な場合に `Jax` を使用します。`Keras`は、一般的なディープラーニングの機能が多く実装されている高水準のフレームワークです。Jaxより少ないコードでモデルを記述することができるため、より複雑なモデルを扱いたい時や、より完成度の高いモデルを見せたい時に使います。もちろん、完全なモデルには`Jax`を使い、詳細な実装は`Keras`で見せることも可能です。これはあくまで原著者がフレームワークを選択した理由です。また、機械学習の導入についての章では`scikit-learn`を扱います。`scikit-learn`は厳密にはディープラーニングではなく古典的な機械学習のパッケージですが、機械学習を用いた問題解決において最も広く用いられているパッケージの一つです。最後に、`Tensorflow`は`Keras`の基盤となるライブラリで、`Keras`で新しいレイヤーを実装したい場合は、`Tensorflow`を用いて実装することになります。`TensorflowProbability` は `Tensorflow` の拡張機能で、深層生成モデルで使用するランダム変数と確率分布をサポートしています。なお、本書では取り扱いませんが、 `PyTorch` は重要なディープラーニングフレームワークの一つです。これは最近、ディープラーニングの研究において最も人気のあるフレームワークとして普及しています。とはいえ根本的に、本書は数式と実装の詳細を紹介することで、フレームワークに依存しない概念を学ぶことができるようにしました。そのため、PyTorchやMXNet、あるいはこれから登場する新しいフレームワークであれ、すぐに本書のアイディアを実装できるはずです。

私が見る学生の最もよくある間違いの1つは、Webで質問を検索したり、ディープラーニングフレームワークのドキュメントを読んだりして、ディープラーニングを学ぼうとすることです。*これは、ディープラーニングを学ぶには最悪の方法です。* 世の中には多くの情報がありますが、こうしてしまうとディープラーニングについて歪んだ、フレームワーク特有の理解をしてしまうことになります。検索結果の上位にあるものは、関連性が高く人気があるかもしれませんが、それがあなたの学習の助けになるとは限らないことを覚えておいてください。さらに重要なのは、ブログやStack overflowでディープラーニングを学ぶと、数学と直感を把握するのがとても難しくなることです。Web検索やコードのハッキングは、（良くも悪くも）間違いなくディープラーニングの一部ですが、これは、実装したいモデルの背後にある数学と詳細をしっかり把握した後で行うべきものです。

## Interactivity

On each chapter, you'll see the &nbsp;<i aria-label="Launch interactive content" class="fas fa-rocket"></i>&nbsp; button on the top. This launches the chapter as an interactive Google Colab. Each chapter also includes notes on the packages that may need to be installed. If you have problems with install, the complete current list of packages for the textbook is available [here](https://github.com/whitead/dmol-book/blob/master/package/requirements.txt).

When using interactivity, many of the chapter will benefit from using a graphics processing unit (GPU). GPUs are what makes deep learning fast enough to be practical on large dataset. This is possible in Google Colab, but may require additional steps if running this locally. Check the documentation of the package you're using (e.g., `Jax`, `PyTorch`, `Tensorflow`) to find out how to use a GPU locally. I have carefully constructed the examples to be small enough though to run on a normal CPU in a laptop though, so the GPU is optional.


## Example models

Here are the major models we will construct learn to implement in this book: *Sorry, but I'm unable to link directly to the examples, so you'll need to scroll to them.*

1. We explore predicting solubility of molecules with {doc}`graph convolutional neural networks<dl/gnn>`, {doc}`recurrent neural networks<dl/NLP>`, {doc}`dense neural networks<dl/xai>`, and {doc}`kernel learning<ml/kernel>`.
2. We implement a SchNet model to {doc}`predict what space group a structure belongs to<dl/gnn>`.
3. We implement a Recurrent Neural Network to {doc}`predict the solubility of proteins/peptides<dl/layers>` and {doc}`predict the if a peptide will lyse red blood cells<dl/xai>`.
4. We predict the DFT single-point energy of molecules {doc}`with a graph convolutional neural network<applied/QM9>`.
5. We propose new molecules with a generative {doc}`recurrent neural network<applied/MolGenerator>`.
6. We learn to align and embed polymer trajectories with {doc}`variational autoencoders<dl/VAE>` and {doc}`equivariant data representations<dl/data>`.
7. We classify if molecules are likely to be toxic with {doc}`logistic regression<ml/classification>`.

and there are many smaller examples throughout the book.

## Table of Contents

```{tableofcontents}
```

## Contributors

Thank you to contributors for offering suggestions, identifying errors, and helping improve this book! Twitter handles, if available

### Contributed Chapter

1. Mehrad Ansari (@MehradAnsari)

### Contributed Content to Chapter

1. Geemi Wellawatte (@GWellawatte)

### Substantial Feedback on Content

1. Lily Wang (@lilyminium)
2. Marc Finzi (@m_finzi)
3. Kevin Jablonka (@kmjablonka)
4. Elana Simon
5. Cathrine Bergh (@cathrinebergh)

### Code Fixes, Math Fixes, Language Fixes

1. Oion Akif
2. Heta Gandhi (@gandhi_heta)
3. Mattias Hartveit
4. Andreas Krämer
5. Mehrad Ansari (@MehradAnsari)
6. Ritsuya Niwayama
7. Varsha Jain
8. Simon Duerr
9. Julia Westermayr (@JWestermayr)
10. Ernest Awoonor-Williams
11. Joshua Schrier (@joshuaschrier)
12. Marin Bukov
13. Arun Pa Thiagarajan (@arunppsg)
14. Ankur Parmar
15. Erik Thiede (@erik_der_elch)

## Citation

Please cite the [livecommsj overview article](https://doi.org/10.33011/livecoms.3.1.1499):

```bibtex
@article{white2021deep,
  title={Deep Learning for Molecules and Materials},
  journal={Living Journal of Computational Molecular Science},
  author={White, Andrew D},
  url={https://dmol.pub},
  year={2021},
  volume={3},
  number={1},
  pages={1499},
  doi={10.33011/livecoms.3.1.1499}
}
```

## Funding Support

Research reported in this work was supported by the National Institute of General Medical Sciences of the National Institutes of Health under award number R35GM137966. This material is based upon work supported by the National Science Foundation under Grant No. 1764415.

## License (CC BY-NC 3.0)

Creative Commons Legal Code

Attribution-NonCommercial 3.0 Unported

See complete description of license at [https://creativecommons.org/licenses/by-nc/3.0/](https://creativecommons.org/licenses/by-nc/3.0/) or at repo [https://github.com/whitead/dmol-book](https://github.com/whitead/dmol-book)
