#!/bin/bash

# 遍历文件夹中的所有文件
# フォルダ内のすべてのファイルに対してループ
# Loop through all files in the folder
for FILE in configs/*; do
    # 对每个文件运行 Python 脚本
    # 各ファイルに対してPythonスクリプトを実行
    # Run the Python script for each file
    python main.py -config_path "requirements.txt"
done
