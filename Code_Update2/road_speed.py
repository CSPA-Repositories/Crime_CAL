SPEED_KMH = {
    '一级': 100,
    '二级': 80,
    '三级': 40,
    '四级': 30,
    '高速': 120,
    '国道': 80,
    '省道': 80,
    '县道': 60,
    '乡道': 40,
}
DEFAULT_SPEED_KMH = 30
CONNECTOR_SPEED_KMH = 30
def travel_time_min(length_m, speed_kmh):
    return length_m / (speed_kmh * 1000 / 60)
