import subprocess
import sys
import os

def main():
    # 构造要执行的命令和参数
    cmd = [sys.executable, 'main_track.py', '-cf', './config_track.json']

    # 执行命令
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
