# ──────────────────────────────────────────────────────────────────────────────
# Ui/utils.py
# UI 표시를 위한 유틸리티 함수 모음 (용량 변환, 시간 변환 등)
# ──────────────────────────────────────────────────────────────────────────────
from typing import Optional
import math

def human_bytes(n: Optional[int]) -> str:
    """
    바이트(Byte) 단위의 숫자를 사람이 읽기 쉬운 문자열(KB, MB, GB 등)로 변환합니다.
    
    :param n: 변환할 바이트 수 (정수 또는 None)
    :return: 변환된 문자열 (예: "1.50 MB")
    """
    if n is None:
        return "-"
    
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    size = float(n)
    
    # 1024 단위로 나누며 단위를 변경
    while size >= 1024 and i < len(units) - 1:
        size /= 1024.0
        i += 1
        
    return f"{size:.2f} {units[i]}"


def human_time(seconds: Optional[float]) -> str:
    """
    초(Second) 단위의 시간을 사람이 읽기 쉬운 문자열(시:분:초)로 변환합니다.
    
    :param seconds: 변환할 시간 (초 단위, 실수 또는 None)
    :return: 변환된 문자열 (예: "1h 05m 30s", "45s")
    """
    if seconds is None or seconds < 0 or math.isinf(seconds):
        return "-"
    
    # 시, 분, 초 계산
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m:d}m {s:02d}s"
    return f"{s:d}s"
