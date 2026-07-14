from PIL import Image, ImageDraw, ImageFont
import os

w, h = 1600, 900
img = Image.new('RGB', (w, h), '#07111F')
draw = ImageDraw.Draw(img)

# background gradient-like effect
for y in range(h):
    r = int(7 + (y / h) * 18)
    g = int(17 + (y / h) * 35)
    b = int(31 + (y / h) * 40)
    draw.line([(0, y), (w, y)], fill=(r, g, b))

# glow shapes
for x, y, r, color in [(1180, 180, 220, (34, 146, 255)), (280, 700, 260, (0, 212, 255))]:
    for radius in range(r, 0, -20):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color + (max(0, 80 - radius // 2),))

# title + subtitle
try:
    font_large = ImageFont.truetype('arial.ttf', 72)
    font_mid = ImageFont.truetype('arial.ttf', 30)
    font_small = ImageFont.truetype('arial.ttf', 24)
except Exception:
    font_large = ImageFont.load_default()
    font_mid = ImageFont.load_default()
    font_small = ImageFont.load_default()

draw.text(((w - 700) // 2, 220), 'RAG-Based Chatbot', fill='white', font=font_large)
draw.text(((w - 900) // 2, 330), 'Built with LangChain for intelligent, document-aware conversations', fill='#A9D7FF', font=font_mid)

# feature pills
for idx, pill in enumerate(['RAG', 'LangChain', 'LLM Integration']):
    x = 470 + idx * 240
    y = 430
    bbox = draw.textbbox((0, 0), pill, font=font_small)
    pw = bbox[2] - bbox[0] + 36
    ph = bbox[3] - bbox[1] + 16
    draw.rounded_rectangle((x, y, x + pw, y + ph), radius=16, fill=(255, 255, 255, 40))
    draw.text((x + 18, y + 8), pill, fill='white', font=font_small)

# bottom line
line = 'Smart Answers from Your Data'
draw.text(((w - 500) // 2, 720), line, fill='#7EE7FF', font=font_mid)

# simple AI-like graphic
shape = [(980, 520), (1020, 470), (1070, 500), (1100, 450), (1160, 520), (1100, 580), (1070, 530), (1020, 560)]
draw.polygon(shape, fill=(18, 132, 255))

out_path = os.path.join(os.getcwd(), 'upwork_cover.jpg')
img.save(out_path)
print(out_path)
