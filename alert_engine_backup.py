from dataclasses import dataclass, field
from time import time

CONFIDENCE_THRESHOLD = 0.50
REQUIRED_CONFIRMATIONS = 3
COOLDOWN_SECONDS = 5.0
UNKNOWN_LABEL = "other_unknown"

SEVERITY = {
    "baby_crying": "high",
    "door_knocking": "medium",
    "emergency_siren": "high",
    "footsteps": "medium",
    "glass_breaking": "critical",
    "vehicle_horn": "high",
    "other_unknown": "none",
}

ACTION = {
    "baby_crying": "Check the nearby area or respond to the baby.",
    "door_knocking": "Check the door when convenient.",
    "emergency_siren": "Check surroundings and seek safety if required.",
    "footsteps": "Check surroundings if the event is unexpected.",
    "glass_breaking": "Move away from the area and seek assistance.",
    "vehicle_horn": "Check traffic and surroundings immediately.",
    "other_unknown": "Continue monitoring; no immediate alert.",
}


@dataclass
class AlertDecisionEngine:
    threshold: float = CONFIDENCE_THRESHOLD
    required_confirmations: int = REQUIRED_CONFIRMATIONS
    cooldown_seconds: float = COOLDOWN_SECONDS
    candidate_label: str | None = None
    consecutive_count: int = 0
    last_alert_label: str | None = None
    last_alert_time: float | None = None
    history: list = field(default_factory=list)

    def process(self, label: str, confidence: float, now: float | None = None):
        if now is None:
            now = time()

        result = {
            "label": label,
            "confidence": confidence,
            "status": "NO_ALERT",
            "reason": "",
            "severity": SEVERITY.get(label, "none"),
            "action": ACTION.get(label, "Continue monitoring; no immediate alert."),
            "consecutive_count": 0,
        }

        if label == UNKNOWN_LABEL:
            self.candidate_label = None
            self.consecutive_count = 0
            result["reason"] = "Unknown/background event is excluded from alerts."
            self.history.append(result)
            return result

        if confidence < self.threshold:
            self.candidate_label = None
            self.consecutive_count = 0
            result["reason"] = (
                f"Confidence {confidence:.2f} is below threshold "
                f"{self.threshold:.2f}."
            )
            self.history.append(result)
            return result

        if label == self.candidate_label:
            self.consecutive_count += 1
        else:
            self.candidate_label = label
            self.consecutive_count = 1

        result["consecutive_count"] = self.consecutive_count

        if self.consecutive_count < self.required_confirmations:
            result["status"] = "CANDIDATE"
            result["reason"] = (
                f"Waiting for confirmation {self.consecutive_count}/"
                f"{self.required_confirmations}."
            )
            self.history.append(result)
            return result

        cooldown_active = (
            self.last_alert_label == label
            and self.last_alert_time is not None
            and now - self.last_alert_time < self.cooldown_seconds
        )

        if cooldown_active:
            result["status"] = "SUPPRESSED_COOLDOWN"
            result["reason"] = (
                f"Duplicate {label} alert suppressed during the "
                f"{self.cooldown_seconds:.0f}-second cooldown."
            )
            self.history.append(result)
            return result

        self.last_alert_label = label
        self.last_alert_time = now
        result["status"] = "FINAL_ALERT"
        result["reason"] = (
            f"{self.required_confirmations} consecutive predictions confirmed."
        )
        self.history.append(result)
        return result


if __name__ == "__main__":
    engine = AlertDecisionEngine()

    demo_sequence = [
        ("emergency_siren", 0.82, 0.0),
        ("emergency_siren", 0.76, 1.0),
        ("emergency_siren", 0.88, 2.0),
        ("emergency_siren", 0.91, 3.0),
        ("other_unknown", 0.95, 4.0),
        ("glass_breaking", 0.99, 5.0),
        ("glass_breaking", 0.98, 6.0),
        ("glass_breaking", 0.97, 7.0),
        ("glass_breaking", 0.96, 8.0),
    ]

    print("DeafGuard alert-engine demonstration\n")
    for label, confidence, timestamp in demo_sequence:
        decision = engine.process(label, confidence, timestamp)
        print(
            f"{timestamp:>4.1f}s | {label:<18} | "
            f"{confidence:.0%} | {decision['status']:<20} | "
            f"{decision['reason']}"
        )
