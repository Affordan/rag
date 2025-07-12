"""
终端输出工具函数
使用colorama库提供彩色终端输出
"""

from colorama import init, Fore, Back, Style
import sys

# 初始化colorama
init(autoreset=True)

def print_info(message):
    """打印信息消息（蓝色）"""
    print(f"{Fore.BLUE}[INFO]{Style.RESET_ALL} {message}")

def print_success(message):
    """打印成功消息（绿色）"""
    print(f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} {message}")

def print_warning(message):
    """打印警告消息（黄色）"""
    print(f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} {message}")

def print_error(message):
    """打印错误消息（红色）"""
    print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {message}")

def print_loading(message):
    """打印加载消息（青色）"""
    print(f"{Fore.CYAN}[LOADING]{Style.RESET_ALL} {message}")

def print_step(step_num, message):
    """打印步骤消息（品红色）"""
    print(f"{Fore.MAGENTA}[STEP {step_num}]{Style.RESET_ALL} {message}")

def print_banner(title):
    """打印横幅标题"""
    banner = "=" * 60
    print(f"{Fore.CYAN}{banner}")
    print(f"{Fore.CYAN}{title.center(60)}")
    print(f"{Fore.CYAN}{banner}{Style.RESET_ALL}")

def print_section(section_name):
    """打印章节标题"""
    print(f"\n{Fore.YELLOW}{'='*10} {section_name} {'='*10}{Style.RESET_ALL}")

def print_file_operation(operation, filename):
    """打印文件操作消息"""
    print(f"{Fore.GREEN}[FILE]{Style.RESET_ALL} {operation}: {Fore.WHITE}{filename}{Style.RESET_ALL}")

def print_network(message):
    """打印网络相关消息（青色）"""
    print(f"{Fore.CYAN}[NETWORK]{Style.RESET_ALL} {message}")

def print_database(message):
    """打印数据库相关消息（品红色）"""
    print(f"{Fore.MAGENTA}[DATABASE]{Style.RESET_ALL} {message}")

def print_model(message):
    """打印模型相关消息（蓝色+亮色）"""
    print(f"{Fore.BLUE}{Style.BRIGHT}[MODEL]{Style.RESET_ALL} {message}")

def print_server(message):
    """打印服务器相关消息（绿色+亮色）"""
    print(f"{Fore.GREEN}{Style.BRIGHT}[SERVER]{Style.RESET_ALL} {message}")
