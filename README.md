<a href='https://github.com/Junwu0615/LCII-Rec-Model'><img alt='GitHub Views' src='https://views.whatilearened.today/views/github/Junwu0615/LCII-Rec-Model.svg'> 
<a href='https://github.com/Junwu0615/LCII-Rec-Model'><img alt='GitHub Clones' src='https://img.shields.io/badge/dynamic/json?color=success&label=Clone&query=count&url=https://gist.githubusercontent.com/Junwu0615/7f654406c51d568d31d565347f22d609/raw/LCII-Rec-Model_clone.json&logo=github'> </br>
[![](https://img.shields.io/badge/Project-RecommendationSystem-blue.svg?style=plastic)](https://pse.is/5blcdx) 
[![](https://img.shields.io/badge/Project-Tensorflow-blue.svg?style=plastic)](https://pypi.org/project/tensorflow/) 
[![](https://img.shields.io/badge/Language-Python-blue.svg?style=plastic)](https://www.python.org/) </br>
[![](https://img.shields.io/badge/Package-Numpy-green.svg?style=plastic)](https://pypi.org/project/numpy/) 
[![](https://img.shields.io/badge/Package-Pandas-green.svg?style=plastic)](https://pypi.org/project/pandas/) 
[![](https://img.shields.io/badge/Package-Matplotlib-green.svg?style=plastic)](https://pypi.org/project/matplotlib/) 
[![](https://img.shields.io/badge/Model-LCII-red.svg?style=plastic)](https://github.com/Junwu0615/LCII-Rec-Model) 
[![](https://img.shields.io/badge/Model-II_RNN-red.svg?style=plastic)](https://github.com/olesls/master_thesis) 


## [Latent-Context II Gated Recurrent Unit ( LCII ) :<br/>運用潛在互動訊息於遞歸神經網路建構之階段感知式推薦](https://pse.is/5blcdx)

> 　　推薦系統從協同過濾等傳統推薦方法，到近年盛行的類神經網路方法，都致力於提升推薦之效能。
> 而階段感知式推薦能解決階段式推薦所無法達成的個性化推薦，也因此使其成為推薦系統領域的重要研究之一。
> 最近的研究大多是基於遞歸神經網路模型或增添注意力機制，並以階段內或階段間的內容，作為改善推薦準確率的方法。
> 然而，這些研究並未將互動紀錄彼此間的潛在訊息完全運用於模型的建置中。
> 此外，將互動紀錄嚴苛地界定長短期偏好範圍會使模型的學習失準。
> 因此，本研究將採用雙層門控循環單元，加入彈性界定長短期偏好之設計並運用潛在互動訊息配合注意力機制，以階段感知式之推薦進行研究拓展。<br/>
> 　　本研究模型中的內層神經循環網路主要作用於提取潛在內容訊息之特徵，將該訊息與當下互動紀錄進行融合，以獲得更好的推薦輔助訊息;
> 模型中的外層神經循環網路在特徵學習有二種策略 : 單一特徵融合方式或是額外加入長短期偏好特徵融合的方式，另外後者又可依融合時機再分成前組合及後組合兩類。
> 此外，融合方式也以固定或學習比例而有不同的設計。我們依此稱提出的方法分別為LCII-Pre(前組合)與 LCII-Post(後組合)，並以下標Fix或LP指出該設計。
> 我們進行大量的實驗來評估所提方法的效能，並與近年知名推薦方法於公開的資料集進行分析比較。
> 實驗結果顯示，在 MovieLens 1M的Recall@20中，LCII-Pre<sub>Fix</sub> 的效能分別比II-RNN以及BERT4Rec<sub>+ST+TSA</sub> 高 1.85% 和 2.54%;
> 於Steam的Recall@20中，LCII-Post<sub>LP</sub> 的效能分別比 II-RNN 以及 BERT4Rec<sub>+ST+TSA</sub> 提升了18.66% 和5.5%;
> 而在 Amazon的Recall@20 中 LCII 的效能比II-RNN以及 CAII分別多了2.59% 和1.89%。
> 經由實驗表明，本研究所提出之運用潛在互動訊息與長短期偏好結合之遞歸神經網路方法，能有效提升推薦系統的效能。


## STEP.1　CLONE 

```code
git clone https://github.com/Junwu0615/LCII-Rec-Model.git
```

## STEP.2　DOWNLOADS DATASETS

於 `/Datasets` 將資料集放入，各資料集來源如下所示 :<br/>
[Amazon](http://jmcauley.ucsd.edu/data/amazon) / [Last.fm / Tmall](https://github.com/RUCAIBox/RecSysDatasets) / [Steam](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data) / [MovieLens_1M / MovieLens_20M](https://grouplens.org/datasets/movielens/) / [Reddit / Instacart](https://github.com/olesls/master_thesis)

## STEP.3　PREPROCESS

於 `/Preprocessing` 進行資料預處理。<br/>
- `amazon_preprocess_ver. BERT4Rec.ipynb` : BERT4Rec 的 Amazon 輸入。<br/>
- `amazon_preprocess_ver. LCII.ipynb` : LCII 的 Amazon 輸入。<br/>
- `ml_steam_preprocess_ver. LCII.ipynb` : LCII 的 MovieLens 1M 和 Steam 輸入。<br/>
- `see_pickle_...ipynb` : 檢視預處理完的內容。

## STEP.4　HYPER PARAMETERS

至 `LCII_hyper_parameters_preference.py` 設置超參數。<br/>
#需將超參數檔案開啟，進行編寫設置。<br/>

 - Dataset　`reddit` / `lastfm` / `instacart` / `tmall` / `steam` / `MovieLens_1M` / `MovieLens_20M`
 - Switch Plot　`sum` / `dot` / `attention_gate_sum` / `attention_gate_dot`
 - Switch Initial State　`True` / `False`
 - Fusion Way　`att` / `lp` / `fix` / `none`
 - Strategy　`pre-combine` / `post-combine` / `original`
 - Window　`0-100`
 - Long Score　`0.0-1.0` / `'no_use'`
 - Short Score　`0.0-1.0` / `'no_use'`
 - Embedding Size　`30` / `50` / `80` / `100` / `200` / `300` / `500` / `800` / `1000`
 - Batch Size　`16` / `32` / `64` / `100` / `128` / `256` / `512`
 - Learning Rate　`0.001` / `0.01` / ...
 - Dropout　`0.8`
 - Max Epoch　`100` / `200` / ...
 - Threshold　`98` #recall@5 如果超過 98 就判定 overfitting
 - 是否要在input_sum 加入FC　`True` / `False`<br/>


## STEP.5　RUN MAIN PROGRAM

執行主程式 `LCII Integrated Model.ipynb`。<br/>
- 執行完畢後會在 `/Testlog` 產出實驗結果 ( .txt / .png )。
