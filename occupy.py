import os
import time
import argparse
from tqdm import tqdm

def get_timestamp_suffix():
    """获取当前时间戳的后7位"""
    # 获取当前时间戳（整数秒）
    ts = int(time.time())
    # 转为字符串并取后7位
    return str(ts)[-7:]

def generate_files(target_dir, num_files, file_size_gb=1):
    # 1. 确保目标目录存在
    if not os.path.exists(target_dir):
        try:
            os.makedirs(target_dir)
            print(f"✅ 已创建目录: {target_dir}")
        except OSError as e:
            print(f"❌ 创建目录失败: {e}")
            return

    # 2. 计算字节数和配置缓冲区
    file_size_bytes = int(file_size_gb * 1024 * 1024 * 1024)
    chunk_size = 10 * 1024 * 1024  # 10MB 缓冲区
    
    # 生成随机数据块 (只生成一次，重复使用以提高速度)
    # 如果完全不在意内容只在意速度，可用 b'\0' * chunk_size 代替
    buffer_data = os.urandom(chunk_size)

    print(f"🚀 开始任务: 目标目录 '{target_dir}' | 生成 {num_files} 个文件 | 单个大小 {file_size_gb}GB")
    print("-" * 50)

    for i in range(num_files):
        # 3. 生成文件名 (model_时间戳后7位.pth)
        # 注意：如果写入速度极快(小于1秒)，时间戳可能重复。
        # 这里加一个简单的校验，如果文件已存在，稍微等待一下更新时间戳
        while True:
            time_suffix = get_timestamp_suffix()
            filename = f"model_{time_suffix}.pth"
            file_path = os.path.join(target_dir, filename)
            if not os.path.exists(file_path):
                break
            time.sleep(1) # 等待1秒以获得新的时间戳

        # 4. 写入文件并显示进度条
        try:
            # desc 设置进度条左侧的文字描述
            with open(file_path, 'wb') as f:
                with tqdm(total=file_size_bytes, unit='B', unit_scale=True, unit_divisor=1024, 
                          desc=f"[{i+1}/{num_files}] {filename}", ncols=100) as pbar:
                    
                    bytes_written = 0
                    while bytes_written < file_size_bytes:
                        remaining = file_size_bytes - bytes_written
                        current_chunk_size = min(chunk_size, remaining)
                        
                        # 切片buffer以适应最后一块数据
                        if current_chunk_size == chunk_size:
                            f.write(buffer_data)
                        else:
                            f.write(buffer_data[:current_chunk_size])
                            
                        bytes_written += current_chunk_size
                        pbar.update(current_chunk_size)
                        
        except OSError as e:
            print(f"\n❌ 写入出错 (可能是磁盘空间不足): {e}")
            break
        except KeyboardInterrupt:
            print("\n⚠️ 用户手动中断任务")
            break

    print("-" * 50)
    print("✨ 所有任务已结束。")

def main():
    # 5. 设置 Argparse 命令行参数
    parser = argparse.ArgumentParser(description="循环生成指定大小的 .pth 文件工具")
    
    parser.add_argument('--dir', '-d', type=str, required=True, 
                        help='目标文件夹路径 (例如: ./data)')
    parser.add_argument('--num', '-n', type=int, required=True, 
                        help='需要生成的文件数量 (例如: 5)')
    
    # 解析参数
    args = parser.parse_args()
    
    # 执行主逻辑
    generate_files(args.dir, args.num)

if __name__ == "__main__":
    main()