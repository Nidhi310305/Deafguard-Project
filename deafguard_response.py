from dataclasses import dataclass


@dataclass
class AlertResponse:
    event: str
    confidence: float
    severity: str
    environment: str
    message: str
    symbol: str
    vibration_pattern: str
    priority: int


EVENT_POLICIES = {
    "emergency_siren": {
        "severity": "HIGH",
        "environment": "outdoor",
        "message": "Emergency siren detected. Check your surroundings.",
        "symbol": "emergency_siren.png",
        "vibration_pattern": "three_short_pulses",
        "priority": 3,
    },
    "vehicle_horn": {
        "severity": "HIGH",
        "environment": "outdoor",
        "message": "Vehicle horn detected. Check nearby traffic.",
        "symbol": "vehicle_horn.png",
        "vibration_pattern": "three_short_pulses",
        "priority": 3,
    },
    "glass_breaking": {
        "severity": "CRITICAL",
        "environment": "indoor_outdoor",
        "message": "Possible glass breaking detected. Move away and seek assistance.",
        "symbol": "glass_breaking.png",
        "vibration_pattern": "long_short_long",
        "priority": 4,
    },
    "baby_crying": {
        "severity": "HIGH",
        "environment": "indoor",
        "message": "Baby crying detected. Check the nearby area.",
        "symbol": "baby_crying.png",
        "vibration_pattern": "three_short_pulses",
        "priority": 3,
    },
    "door_knocking": {
        "severity": "MEDIUM",
        "environment": "indoor",
        "message": "Someone may be at the door.",
        "symbol": "door_knocking.png",
        "vibration_pattern": "two_short_pulses",
        "priority": 2,
    },
    "footsteps": {
        "severity": "MEDIUM",
        "environment": "indoor_outdoor",
        "message": "Footsteps detected. Check your surroundings if unexpected.",
        "symbol": "footsteps.png",
        "vibration_pattern": "two_short_pulses",
        "priority": 2,
    },
    "other_unknown": {
        "severity": "NONE",
        "environment": "uncertain",
        "message": "Unknown or background sound.",
        "symbol": "unknown_sound.png",
        "vibration_pattern": "none",
        "priority": 0,
    },
}


def create_alert_response(event, confidence):
    policy = EVENT_POLICIES.get(
        event,
        {
            "severity": "NONE",
            "environment": "uncertain",
            "message": "Uncertain sound detected.",
            "symbol": "uncertain.png",
            "vibration_pattern": "none",
            "priority": 0,
        },
    )

    return AlertResponse(
        event=event,
        confidence=round(float(confidence), 4),
        severity=policy["severity"],
        environment=policy["environment"],
        message=policy["message"],
        symbol=policy["symbol"],
        vibration_pattern=policy["vibration_pattern"],
        priority=policy["priority"],
    )


if __name__ == "__main__":
    response = create_alert_response("emergency_siren", 0.94)
    print(response)
