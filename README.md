<a href='https://github.com/Junwu0615/LCII-Rec-Model'><img alt='GitHub Views' src='https://views.whatilearened.today/views/github/Junwu0615/LCII-Rec-Model.svg'> 
<a href='https://github.com/Junwu0615/LCII-Rec-Model'><img alt='GitHub Clones' src='https://img.shields.io/badge/dynamic/json?color=success&label=Clone&query=count_total&url=https://gist.githubusercontent.com/Junwu0615/7f654406c51d568d31d565347f22d609/raw/LCII-Rec-Model_clone.json&logo=github'> </br>
[![](https://img.shields.io/badge/Project-RecommendationSystem-blue.svg?style=plastic)](https://pse.is/5jtztg) 
[![](https://img.shields.io/badge/Project-Tensorflow_GPU_2.4.0-blue.svg?style=plastic)](https://pypi.org/project/tensorflow/) 
[![](https://img.shields.io/badge/Language-Python_3.8.18-blue.svg?style=plastic)](https://www.python.org/) </br>
[![](https://img.shields.io/badge/Package-Numpy_1.20.3-green.svg?style=plastic)](https://pypi.org/project/numpy/) 
[![](https://img.shields.io/badge/Package-Pandas_1.5.3-green.svg?style=plastic)](https://pypi.org/project/pandas/) 
[![](https://img.shields.io/badge/Package-Matplotlib_3.7.4-green.svg?style=plastic)](https://pypi.org/project/matplotlib/) 
[![](https://img.shields.io/badge/Model-LCII_v1.1-red.svg?style=plastic)](https://github.com/Junwu0615/LCII-Rec-Model) 
[![](https://img.shields.io/badge/Model-II_RNN-red.svg?style=plastic)](https://github.com/olesls/master_thesis) 

## 歷史紀錄
| 事件 | 敘述 | 時間 |
| :--: | :-- | :--: |
| Master's thesis | 推薦系統模型 : [LCII-Rec-Model](https://pse.is/5jtztg) | 2023/01/19 |
| LCII v1.0 | 第一次將程式碼發布於 GitHub | 2023/12/09 |
| LCII v1.1 | 精簡程式碼，以及 LCII 環境安裝說明 | 2024/02/14 |

</br>

## HOW TO USE
### STEP.1　CREATE ENVIRONMENT
本論文程式碼是建立於 Tensorflow 1.14.0，但如今 TF 版本和顯卡設備日新月異的現在，若沒有正確地方式啟動本模型，LCII 將無法成功運作，因此我微調了一下程式碼，讓架構不至於需要重構。
- 當初程式是在 Anaconda 環境下完成，我將可運行之 conda 環境匯出成 [yaml](/environment.yaml) 檔。
  - `conda env create -f C:\Users\xxx\LCII-Rec-Model\environment.yaml` (自行設路徑安裝)。
  ```py
  # xxx > username
  prefix: C:\Users\xxx\anaconda3\envs\lcii
  ```
- N 卡安裝 TF-GPU 環境網路教學很多，我[推薦一篇](https://zhuanlan.zhihu.com/p/612864973)。
- 我運行環境如下所示 :
  - Tensorflow-GPU 2.4.0
    - Python : 3.8.18
    - 顯示卡 : Nvidia GeForce GTX 4070 | 驅動 : 551.23
    - CUDA : 11.0
    - cuDNN : 8.0
- 程式碼已改寫成 TF 2.0 以上版本皆可運行的內容。另外想抱怨建置環境這塊，它 ! 非 ! 常 ! 麻 ! 煩 !，我也是額外多買了一塊 N 卡 (GTX 4070)，解決原先 A 卡 (RX 6090 XT) 在 win 環境不支援等問題，且又花了快 2 天時間才建置好運行環境...
- 大概簡述安裝流程 : 
  - 顯卡若是 N 卡的話，去 [Nvidia 官網](https://www.nvidia.com.tw/Download/index.aspx?lang=tw)安裝對應持有顯卡的驅動程式。
  - [查看持有顯卡驅動版本](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)，cmd 輸入 : `nvidia-smi`，確認符合需求的 CUDA 版本。
    - 由於 RTX 30 系列之後的 GPU 不支援 CUDA 10.0 以下(含)，因此我也只能裝 CUDA 11.0 以上版本 (有實測撞牆過...)。
  - 接著查閱 [Tensorflow-GPU 版本](https://www.tensorflow.org/install/source_windows?hl=zh-tw)，已確定需要的 CUDA / cuDNN 版本。
  - 下載各自對應的 [CUDA](https://developer.nvidia.com/cuda-toolkit-archive) / [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) 版本。
    - CUDA 自定義安裝並勾選"只安裝 CUDA"即可 (**安裝過程會將系統環境變數新增上去**)。另外把 cuDNN 包解壓縮後，將內容物 (`bin` / `include` / `lib`) 剪下貼至 CUDA 安裝地。

</br>

### STEP.2　CLONE 
```py
git clone https://github.com/Junwu0615/LCII-Rec-Model.git
```
跳至 **LCII-Rec-Model** 專案目錄 : `cd LCII-Rec-Model`

</br>

### STEP.3　DOWNLOADS DATASETS
跳至 **preprocess** 目錄 : `cd preprocess`</br>

#### 資料集來源 :
| Dataset | Search the Page for Keywords |
| :-- | :-- |
| [Amazon](https://nijianmo.github.io/amazon/index.html) | `Clothing Shoes and Jewelry \| 5-core (11,285,464 reviews) \| 1.17 GB` |
| [MovieLens-1M](https://grouplens.org/datasets/movielens/) | `ml-1m.zip \| 6 MB` |
| [Steam](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data) | `Version 2: Review Data \| 1.3 GB` |
| [LCII Dataset Backup](https://drive.google.com/drive/folders/1itPZi2Ae0p2HDGSfPVCuxqcJdOo-WyVF?usp=sharing) | `Dataset has been backup to the cloud` |

#### 下載資料集並解壓縮到指定位置 :
| Dataset | Path |
| :-- | :-- |
| Amazon | 需將 `Clothing_Shoes_and_Jewelry_5.json.gz` 放置於 `LCII-Rec-Model/datasets`</br>解壓縮完路徑為: `LCII-Rec-Model/datasets/Clothing_Shoes_and_Jewelry_5.json` |
| MovieLens-1M | 需將 `ml-1m.zip` 放置於 `LCII-Rec-Model/datasets`</br>解壓縮完路徑為: `LCII-Rec-Model/datasets/ml-1m` |
| Steam | 需將 `steam_reviews.json.gz` 放置於 `LCII-Rec-Model/datasets`</br>解壓縮完路徑為: `LCII-Rec-Model/datasets/steam_new.json` |

</br>

### STEP.4　PREPROCESS

#### 資料預處理指令 :
| Dataset | Program Instructions |
| :-- | :-- |
| Amazon | `python amazon_preprocess.py` |
| Amazon ver. BERT4Rec | `python amazon_preprocess_ver_bert4rec.py` |
| MovieLens-1M | `python ml_steam_preprocess.py -d MovieLens-1M` |
| Steam | `python ml_steam_preprocess.py -d Steam` |

![preprocess_ml-1m.gif](/sample_img/preprocess_ml-1m.gif)

#### 查閱預處理後之資料 :
| Dataset | Program Instructions |
| :-- | :-- |
| Amazon | `python see_pickle_amazon.py` |
| MovieLens-1M | `python see_pickle_ml_steam.py -d MovieLens-1M` |
| Steam | `python see_pickle_ml_steam.py -d Steam` |

![see_ml-1m.gif](/sample_img/see_ml-1m.gif)

</br>

### STEP.5　HELP

回到 **LCII-Rec-Model** 專案目錄 `cd ..`
```py
python LCII-Rec-Model.py -h
```
 - `-d`　Dataset :　`Amazon` / `MovieLens-1M` / `Steam`
 - `-sp`　Switch Plot :　`sum` / `dot` / `attention_gate_sum` / `attention_gate_dot`
 - `-sis`　Switch Initial State :　`True` / `False`
 - `-fw`　Fusion Way :　`att` / `lp` / `fix` / `none`
 - `-s`　Strategy :　`pre-combine` / `post-combine` / `original`
 - `-w`　Window :　`0-100` / `'no_use'`
 - `-ls`　Long Score :　`0.0-1.0` / `'no_use'`
 - `-ss`　Short Score :　`0.0-1.0` / `'no_use'`
 - `-es`　Embedding Size :　`30` / `50` / `80` / `100` / `200` / `300` / `500` / `800` / `1000`
 - `-bs`　Batch Size :　`16` / `32` / `64` / `100` / `128` / `256` / `512`
 - `-lr`　Learning Rate :　`0.001` / `0.01` / ...
 - `-dr`　Dropout :　`0.8`
 - `-me`　Max Epoch :　`100` / `200` / ...
 - `-t`　Threshold :　`98`
 - `-add`　Whether to finally add FC to input_sum :　`True` / `False`<br/>

</br>

### STEP.6　RUN MAIN PROGRAM

下列為 LCII 模型在各資料集中的程式運行之指令範例，執行完畢後會在 `/testlog/` 產出實驗結果 ( .txt / .png )。

#### Amazon
```py
python LCII-Rec-Model.py -d Amazon -sp attention_gate_dot -sis True -fw none -s original -w 'no_use' -ls 'no_use' -ss 'no_use' -es 80 -bs 100 -lr 0.001 -dr 0.8 -me 100 -t 98 -add False
```
![amazon.jpg](/sample_img/amazon_result.jpg)

#### MovieLens-1M
```py
python LCII-Rec-Model.py -d MovieLens-1M -sp attention_gate_dot -sis True -fw fix -s pre-combine -w 30 -ls 0.8 -ss 0.2 -es 80 -bs 100 -lr 0.01 -dr 0.8 -me 200 -t 98 -add False
```
![ml-1m.jpg](/sample_img/ml-1m_result.jpg)

#### Steam
```py
python LCII-Rec-Model.py -d Steam -sp attention_gate_dot -sis True -fw fix -s post-combine -w 4 -ls 0.2 -ss 0.8 -es 80 -bs 100 -lr 0.001 -dr 0.8 -me 200 -t 98 -add False
```
![steam.jpg](/sample_img/steam_result.jpg)

</br>

下列為相關 Baseline
- [M. Ruocco, O. S. L. Skrede, and H. Langseth | II-RNN](https://github.com/olesls/master_thesis)
- [J. J. Seol, Y. Ko, and S. G. Lee | BERT4Rec](https://github.com/theeluwin/session-aware-bert4rec)
- most-recent / most-pop / kNN
  - 跳至 **baseline** 目錄 : `cd baseline`</br>
  - 執行指令如下 :
    - Amazon
    ```py
    python Baseline.py -d Amazon
    ```
    - MovieLens-1M
    ```py
    python Baseline.py -d MovieLens-1M
    ```
    - Steam
    ```py
    python Baseline.py -d Steam
    ```