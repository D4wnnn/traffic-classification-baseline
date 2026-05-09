import binascii
from PIL import Image
import scapy.all as scapy
import numpy as np
import re

def clean_packet(packet):
    if packet.haslayer(scapy.Ether):
        packet = packet[scapy.Ether].payload

    if packet.haslayer(scapy.IP):
        packet[scapy.IP].src = "0.0.0.0"
        packet[scapy.IP].dst = "0.0.0.0"
    elif packet.haslayer("IPv6"):
        packet["IPv6"].src = "::"
        packet["IPv6"].dst = "::"
    else:
        # 如果没有IP层，则无法处理，跳过
        return None

    if packet.haslayer(scapy.UDP):
        packet[scapy.UDP].sport = 0
        packet[scapy.UDP].dport = 0
    elif packet.haslayer(scapy.TCP):
        packet[scapy.TCP].sport = 0
        packet[scapy.TCP].dport = 0
    return packet


def get_header_payload(packet):
    packet = clean_packet(packet)
    # 增加一个检查，如果packet为None，则返回空值
    if packet is None:
        return "", ""

    header, payload = "", ""
    if packet.haslayer(scapy.IP):
        header = (binascii.hexlify(bytes(packet["IP"]))).decode()
    elif packet.haslayer("IPv6"):
        header = (binascii.hexlify(bytes(packet["IPv6"]))).decode()

    if packet.haslayer("Raw"):
        payload_bytes = bytes(packet["Raw"])
        # 确保载荷不为空
        if payload_bytes:
            payload = (binascii.hexlify(payload_bytes)).decode()
            # 避免因payload为空导致header被错误清空
            if payload in header:
                header = header.replace(payload, "")

    return header, payload


# def read_MFR_bytes(file_name):
#     packets = scapy.rdpcap(file_name)
#     data = []
#     for packet in packets:
#         if packet.haslayer("IP"):
#             ip_layer = packet["IP"]
#         elif packet.haslayer("IPv6"):
#             ip_layer = packet["IPv6"]
#         else:
#             raise
#         header = (binascii.hexlify(bytes(ip_layer))).decode()
#         try:
#             payload = (binascii.hexlify(bytes(packet["Raw"]))).decode()
#             header = header.replace(payload, "")
#         except:
#             payload = ""
#         if len(header) > 160:
#             header = header[:160]
#         elif len(header) < 160:
#             header += "0" * (160 - len(header))
#         if len(payload) > 480:
#             payload = payload[:480]
#         elif len(payload) < 480:
#             payload += "0" * (480 - len(payload))
#         data.append((header, payload))
#         if len(data) >= 5:
#             break
#     if len(data) < 5:
#         for i in range(5 - len(data)):
#             data.append(("0" * 160, "0" * 480))
#     final_data = ""
#     for h, p in data:
#         final_data += h
#         final_data += p
#     return final_data
def read_MFR_bytes(file_name, mask_shortcuts=True):
    """
    修改：新增 mask_shortcuts 参数。
    """
    packets = scapy.rdpcap(file_name)
    data = []
    for p in packets:
        packet = p.copy() # 操作副本

        # --- 根据参数mask数据包 ---
        if mask_shortcuts:
            packet = clean_packet(packet)
            if packet is None: # clean_packet 找不到IP层会返回None
                continue

        if packet.haslayer("IP"):
            ip_layer = packet["IP"]
        elif packet.haslayer("IPv6"):
            ip_layer = packet["IPv6"]
        else:
            continue # 原来是 raise，改为 continue 更健壮

        header = (binascii.hexlify(bytes(ip_layer))).decode()
        
        # --- 优化：使用更健壮的payload提取 ---
        payload = ""
        if packet.haslayer("Raw"):
            payload_bytes = bytes(packet["Raw"])
            if payload_bytes:
                payload = (binascii.hexlify(payload_bytes)).decode()
                # 原始逻辑：从header中移除payload
                if payload in header:
                    header = header.replace(payload, "")
        # --- 修改结束 ---

        if len(header) > 160:
            header = header[:160]
        elif len(header) < 160:
            header += "0" * (160 - len(header))
        if len(payload) > 480:
            payload = payload[:480]
        elif len(payload) < 480:
            payload += "0" * (480 - len(payload))
        data.append((header, payload))
        if len(data) >= 5:
            break
    if len(data) < 5:
        for i in range(5 - len(data)):
            data.append(("0" * 160, "0" * 480))
    final_data = ""
    for h, p in data:
        final_data += h
        final_data += p
    return final_data

def save_image(file_name, path):
    content = read_MFR_bytes(file_name)
    content = np.array([int(content[i : i + 2], 16) for i in range(0, len(content), 2)])
    fh = np.reshape(content, (40, 40))
    fh = np.uint8(fh)
    im = Image.fromarray(fh)
    im.save(path)


def parse_ipv4_frames(hex_string: str):
    """
    从连续十六进制字符串中提取 IPv4 报文（去掉以太网头）
    返回: [{'ip_header': bytes, 'payload': bytes}, ...]
    """
    # 清理输入
    hex_string = hex_string.replace(" ", "").replace("\n", "").lower()
    try:
        data = bytes.fromhex(hex_string)
    except ValueError:
        return []

    frames = []
    # 查找所有 IPv4 起点 (EtherType 0x0800 后的 0x45 或 0x46)
    pattern = re.compile(b"\x08\x00\x00\x45|\x08\x00\x45")
    matches = [m.start() for m in re.finditer(pattern, data)]

    for i, start in enumerate(matches):
        # 找到下一帧的开始位置
        end = matches[i + 1] if i + 1 < len(matches) else len(data)
        frame_data = data[start:end]

        # 找到IPv4头起始点（45或46）
        idx = frame_data.find(b"\x45")
        if idx == -1:
            continue
        ipv4_bytes = frame_data[idx:]
        if len(ipv4_bytes) < 20:
            continue  # 不足一个IPv4头部，跳过

        # 提取头部长度
        ihl = ipv4_bytes[0] & 0x0F
        header_len = ihl * 4

        # 读取总长度字段
        total_len = int.from_bytes(ipv4_bytes[2:4], "big", signed=False)
        if total_len == 0 or total_len > len(ipv4_bytes):
            total_len = len(ipv4_bytes)

        ip_header = ipv4_bytes[:header_len]
        payload = ipv4_bytes[header_len:total_len]

        frames.append({"ip_header": ip_header, "payload": payload})

    return frames

def save_cstnet_datagram_image(datagram_str, output_path, num_packets=5):
    """
    将CSTNET数据集的datagram字符串转换为40×40图像并保存
    与其他数据集保持一致的格式
    
    Args:
        datagram_str: 十六进制字符串
        output_path: 输出图像路径
        num_packets: packet数量，默认5
    """
    try:
        # 解析IPv4帧
        frames = parse_ipv4_frames(datagram_str)
        
        # 每个packet: 80字节header + 240字节payload = 320字节
        # 5个packet: 5 × 320 = 1600字节 = 40×40
        bytes_per_packet = 320
        header_bytes_per_packet = 80  # 与其他数据集保持一致（对应160 hex chars）
        payload_bytes_per_packet = 240  # 与其他数据集保持一致（对应480 hex chars）
        
        total_bytes = num_packets * bytes_per_packet  # 1600
        
        # 创建全零数组
        image_data = np.zeros(total_bytes, dtype=np.uint8)
        
        # 填充每个packet
        for i in range(num_packets):
            offset = i * bytes_per_packet
            
            if i < len(frames):
                frame = frames[i]
                header_bytes = frame["ip_header"]
                payload_bytes = frame["payload"]
                
                # 填充header（80字节，超出部分截断，不足部分保持为0）
                header_arr = np.frombuffer(header_bytes, dtype=np.uint8)
                if len(header_arr) > header_bytes_per_packet:
                    header_arr = header_arr[:header_bytes_per_packet]
                image_data[offset:offset + len(header_arr)] = header_arr
                
                # 填充payload（240字节，超出部分截断，不足部分保持为0）
                payload_arr = np.frombuffer(payload_bytes, dtype=np.uint8)
                if len(payload_arr) > payload_bytes_per_packet:
                    payload_arr = payload_arr[:payload_bytes_per_packet]
                payload_offset = offset + header_bytes_per_packet
                image_data[payload_offset:payload_offset + len(payload_arr)] = payload_arr
        
        # reshape为40×40（与其他数据集保持一致）
        image_matrix = image_data.reshape(40, 40)
        
        # 转换为PIL Image并保存
        im = Image.fromarray(image_matrix)
        im.save(output_path)
        
    except Exception as e:
        print(f"Error processing CSTNET datagram: {e}")