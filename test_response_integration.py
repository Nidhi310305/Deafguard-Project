from deafguard_response import create_alert_response


test_events = [
    ("emergency_siren", 0.94),
    ("vehicle_horn", 0.86),
    ("glass_breaking", 0.91),
    ("baby_crying", 0.88),
    ("door_knocking", 0.79),
    ("footsteps", 0.72),
    ("other_unknown", 0.65),
]


for event, confidence in test_events:
    response = create_alert_response(event, confidence)

    print("-" * 60)
    print("Event:", response.event)
    print("Confidence:", f"{response.confidence:.2%}")
    print("Severity:", response.severity)
    print("Environment:", response.environment)
    print("Message:", response.message)
    print("Symbol:", response.symbol)
    print("Vibration:", response.vibration_pattern)
    print("Priority:", response.priority)
