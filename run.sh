#!/bin/bash

# フォルダ内のすべてのファイルに対してループ
for FILE in configs/*; do
    # 各ファイルに対してPythonスクリプトを実行
    python main.py -config_path "$FILE"
done