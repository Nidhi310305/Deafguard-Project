from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

output_folder = Path("assets/symbols")
output_folder.mkdir(parents=True, exist_ok=True)

symbols = {
    "emergency_siren": ("SIREN", "#B00020"),
    "vehicle_horn": ("HORN", "#D97706"),
    "glass_breaking": ("GLASS", "#7F1D1D"),
    "baby_crying": ("BABY", "#7C3AED"),
    "door_knocking": ("DOOR", "#1D4ED8"),
    "footsteps": ("STEPS", "#047857"),
    "unknown_sound": ("UNKNOWN", "#4B5563"),
    "uncertain": ("?", "#6B7280"),
}

for filename, (label, color) in symbols.items():
    image = Image.new("RGB", (600, 400), "white")
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle(
        (20, 20, 580, 380),
        radius=30,
        fill=color,
        outline="black",
        width=6,
    )

    try:
        font = ImageFont.truetype("arial.ttf", 58)
    except OSError:
        font = ImageFont.load_default()

    box = draw.textbbox((0, 0), label, font=font)
    text_width = box[2] - box[0]
    text_height = box[3] - box[1]

    draw.text(
        ((600 - text_width) / 2, (400 - text_height) / 2),
        label,
        fill="white",
        font=font,
    )

    image.save(output_folder / f"{filename}.png")

print("Symbol images created successfully.")
