"""
将documents目录下的所有txt文件转换为UTF-8编码

这个脚本会:
1. 检测所有txt文件的原始编码
2. 将内容转换为UTF-8编码
3. 覆盖原始文件
"""

import os
import sys
import chardet
from pathlib import Path

def detect_encoding(file_path):
    """检测文件编码"""
    try:
        with open(file_path, 'rb') as f:
            # 只读取文件的前10000个字节来检测编码，提高效率
            raw_data = f.read(10000)
            result = chardet.detect(raw_data)
            return result['encoding']
    except Exception as e:
        print(f"检测 {file_path} 编码时出错: {e}")
        return None

def convert_file_to_utf8(file_path):
    """将文件转换为UTF-8编码并覆盖原文件"""
    try:
        # 检测原始文件编码
        encoding = detect_encoding(file_path)
        if not encoding or encoding.lower() in ('ascii', 'utf-8'):
            # 如果已经是ASCII或UTF-8编码，则跳过
            print(f"文件 {os.path.basename(file_path)} 已经是UTF-8或兼容编码，无需转换")
            return False
        
        print(f"转换 {os.path.basename(file_path)}: {encoding} -> UTF-8")
        
        # 读取原始文件内容
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            content = f.read()
        
        # 以UTF-8编码写入原文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return True
    except Exception as e:
        print(f"转换文件 {os.path.basename(file_path)} 失败: {e}")
        return False

def convert_all_txt_files(directory):
    """转换目录下所有的txt文件"""
    converted = 0
    skipped = 0
    failed = 0
    
    try:
        # 安装chardet库（如果尚未安装）
        try:
            import chardet
        except ImportError:
            print("正在安装必要的chardet库...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "chardet"])
            import chardet
            print("chardet库安装成功")
    except Exception as e:
        print(f"安装依赖库失败: {e}")
        print("请手动运行: pip install chardet")
        return
    
    # 获取所有txt文件的路径
    txt_files = list(Path(directory).glob("**/*.txt"))
    
    if not txt_files:
        print(f"在 {directory} 目录下未找到任何txt文件")
        return
    
    print(f"找到 {len(txt_files)} 个txt文件")
    
    # 转换每个文件
    for file_path in txt_files:
        # 跳过PDF文件
        if "pdf" in str(file_path).lower():
            print(f"跳过PDF文件: {os.path.basename(file_path)}")
            skipped += 1
            continue
            
        if convert_file_to_utf8(file_path):
            converted += 1
        else:
            failed += 1
    
    # 打印结果
    print("\n转换完成!")
    print(f"成功转换: {converted} 个文件")
    print(f"跳过处理: {skipped} 个文件")
    if failed > 0:
        print(f"转换失败: {failed} 个文件")

if __name__ == "__main__":
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 设置documents目录路径
    documents_dir = os.path.join(script_dir, "documents")
    
    print(f"开始处理目录: {documents_dir}")
    convert_all_txt_files(documents_dir)
