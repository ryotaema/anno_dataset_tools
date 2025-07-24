# anno_dataset_tools

このリポジトリは，アノテーションされたデータセットの形式変換やランダム分割を行うためのスクリプト郡.
学習を回す前のデータの整形ようにまとめました．
(対応形式 : YOLO, Pascal VOC)

## 使用方法

0. (アノテーションを行う) 
1. (必要であればパスの書き換えも)
2. スクリプトを実行

```
#利用例(YOLO -> VOCへの変換)
python3 yolo_to_pascal.py
```

# 各スクリプト

-`yolo_to_pascal.py` :YOLO形式からPascal VOC形式への変換
-`pascal_to_yolo.py` :Pascal VOC形式からYOLO形式への変換
-`split_dataset.py` :データセットを3つ(train,val,test)ように分割する

