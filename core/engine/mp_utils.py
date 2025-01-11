import socket

def get_distributed_init_method(ip: str, port: int) -> str:
    return f"tcp://[{ip}]:{port}" if ":" in ip else f"tcp://{ip}:{port}" #多节点的地址初始化


def get_open_port() -> int:
    #try ipv4
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: #
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        #try ipv6
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
