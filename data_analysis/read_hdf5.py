import os
import h5py
import numpy as np
import glob
from pathlib import Path

def read_hdf5_data(directory):
    """
    读取指定目录中的所有HDF5文件,并打印其中数据的形状信息
    
    Args:
        directory: 包含HDF5文件的目录路径
    """
    # 确保目录路径存在
    if not os.path.exists(directory):
        print(f"目录 {directory} 不存在!")
        return
    
    # 获取目录中所有的.h5文件
    h5_files = glob.glob(os.path.join(directory, "*.h5"))
    
    if not h5_files:
        print(f"在 {directory} 中未找到HDF5文件!")
        return
    
    print(f"在 {directory} 中找到了 {len(h5_files)} 个HDF5文件")
    
    # 遍历每个HDF5文件
    for h5_file in h5_files:
        file_size = os.path.getsize(h5_file)
        print(f"\n分析文件: {os.path.basename(h5_file)}, 大小: {file_size} 字节")
        
        try:
            with h5py.File(h5_file, 'r') as f:
                # 如果文件是空的或只有基础结构
                if len(f.keys()) == 0:
                    print("文件为空或只包含基础结构")
                    continue
                
                # 打印文件结构和内容
                print("文件结构:")
                
                def print_attrs(name, obj):
                    """打印属性"""
                    if len(obj.attrs) > 0:
                        print(f"  {name} 的属性:")
                        for key, value in obj.attrs.items():
                            print(f"    {key}: {value}")
                
                def print_structure(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        try:
                            data_sample = obj[()]
                            print(f"  数据集: {name}, 形状: {obj.shape}, 类型: {obj.dtype}")
                            print_attrs(name, obj)
                        except Exception as e:
                            print(f"  读取数据集 {name} 时出错: {e}")
                    elif isinstance(obj, h5py.Group):
                        print(f"  组: {name}")
                        print_attrs(name, obj)
                
                # 打印根级别的键
                print(f"  根级别键: {list(f.keys())}")
                
                # 递归打印所有项目 有点像异步的操作
                f.visititems(print_structure)
                
        except Exception as e:
            print(f"读取文件 {h5_file} 时出错: {e}")


if __name__ == "__main__":
    # 指定目录路径
    data_dir = os.path.join(os.getcwd(), "data", "PullCube", "motionplanning")
    print(f"尝试读取目录: {data_dir}")
    
    # 读取并分析文件
    read_hdf5_data(data_dir)
