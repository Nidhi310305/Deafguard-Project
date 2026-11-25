import time


VIBRATION_PATTERNS = {
    "none": [],
    "two_short_pulses": [0.15, 0.15],
    "three_short_pulses": [0.15, 0.15, 0.15],
    "long_short_long": [0.60, 0.15, 0.60],
}


def trigger_vibration(pattern_name):
    pattern = VIBRATION_PATTERNS.get(pattern_name, [])

    if not pattern:
        return {
            "pattern": pattern_name,
            "status": "No vibration required",
            "pulses": 0,
        }

    for duration in pattern:
        time.sleep(duration)

    return {
        "pattern": pattern_name,
        "status": "Vibration pattern triggered",
        "pulses": len(pattern),
    }


if __name__ == "__main__":
    print(trigger_vibration("three_short_pulses"))
