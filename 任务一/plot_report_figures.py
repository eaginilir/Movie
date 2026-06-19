"""
Generate report figures for task 1.

The bundled Python environment may not include matplotlib, so this script uses
Pillow for PNG output and reportlab for vector PDF output. The figure labels are
kept in English to avoid LaTeX/PDF font issues across machines.
"""
from __future__ import annotations

import os
from collections import Counter

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

from common import BASE_DIR, load_ratings


FIGURE_DIR = os.path.join(BASE_DIR, "figures")

MODEL_RMSE = [
    ("Baseline", 17.2272),
    ("FunkSVD", 16.6316),
    ("SVD++", 16.9001),
    ("Item-KNN", 19.3244),
    ("Ensemble", 16.5188),
]

ENSEMBLE_WEIGHTS = [
    ("Baseline", 0.20),
    ("FunkSVD", 0.65),
    ("SVD++", 0.15),
]

PALETTE = {
    "blue": "#4E79A7",
    "orange": "#F28E2B",
    "green": "#59A14F",
    "red": "#E15759",
    "purple": "#B07AA1",
    "gray": "#5F6670",
    "grid": "#D8DEE9",
    "text": "#263238",
    "background": "#FFFFFF",
}


def ensure_figure_dir():
    os.makedirs(FIGURE_DIR, exist_ok=True)


def load_font(size: int, bold: bool = False):
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def draw_centered(draw, xy, text, font, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    x = xy[0] - (bbox[2] - bbox[0]) / 2
    y = xy[1] - (bbox[3] - bbox[1]) / 2
    draw.text((x, y), text, font=font, fill=fill)


def save_bar_png(filename, title, labels, values, ylabel, colors_list, value_fmt="{:.2f}", y_min=0):
    width, height = 1100, 680
    margin_left, margin_right = 110, 60
    margin_top, margin_bottom = 105, 120
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    image = Image.new("RGB", (width, height), hex_to_rgb(PALETTE["background"]))
    draw = ImageDraw.Draw(image)
    title_font = load_font(34, bold=True)
    axis_font = load_font(22)
    tick_font = load_font(18)
    value_font = load_font(19, bold=True)
    text_color = hex_to_rgb(PALETTE["text"])
    grid_color = hex_to_rgb(PALETTE["grid"])

    draw.text((margin_left, 35), title, font=title_font, fill=text_color)

    y_max = max(values) * 1.12
    if y_min:
        y_max = max(values) + (max(values) - y_min) * 0.35
    tick_count = 5
    for t in range(tick_count + 1):
        frac = t / tick_count
        value = y_min + frac * (y_max - y_min)
        y = margin_top + plot_h - frac * plot_h
        draw.line((margin_left, y, margin_left + plot_w, y), fill=grid_color, width=1)
        draw.text((28, y - 11), f"{value:.1f}" if y_min else f"{int(value):,}", font=tick_font, fill=text_color)

    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill=text_color, width=2)
    draw.line((margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h), fill=text_color, width=2)

    bar_slot = plot_w / len(labels)
    bar_w = bar_slot * 0.56
    for idx, (label, value) in enumerate(zip(labels, values)):
        x0 = margin_left + idx * bar_slot + (bar_slot - bar_w) / 2
        x1 = x0 + bar_w
        bar_h = (value - y_min) / (y_max - y_min) * plot_h
        y0 = margin_top + plot_h - bar_h
        y1 = margin_top + plot_h
        color = hex_to_rgb(colors_list[idx % len(colors_list)])
        draw.rounded_rectangle((x0, y0, x1, y1), radius=8, fill=color)
        draw.text((x0 + bar_w / 2 - 28, y0 - 30), value_fmt.format(value), font=value_font, fill=text_color)
        draw_centered(draw, (x0 + bar_w / 2, margin_top + plot_h + 32), label, tick_font, text_color)

    draw.text((margin_left + plot_w / 2 - 45, height - 42), ylabel, font=axis_font, fill=text_color)
    image.save(os.path.join(FIGURE_DIR, filename + ".png"), dpi=(160, 160))


def save_bar_pdf(filename, title, labels, values, ylabel, colors_list, value_fmt="{:.2f}", y_min=0):
    width, height = landscape((8.8 * inch, 5.4 * inch))
    path = os.path.join(FIGURE_DIR, filename + ".pdf")
    c = canvas.Canvas(path, pagesize=(width, height))
    c.setTitle(title)

    margin_left, margin_right = 0.85 * inch, 0.35 * inch
    margin_top, margin_bottom = 0.75 * inch, 0.75 * inch
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    x_axis = margin_left
    y_axis = margin_bottom

    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(colors.HexColor(PALETTE["text"]))
    c.drawString(margin_left, height - 0.42 * inch, title)

    y_max = max(values) * 1.12
    if y_min:
        y_max = max(values) + (max(values) - y_min) * 0.35

    c.setStrokeColor(colors.HexColor(PALETTE["grid"]))
    c.setLineWidth(0.6)
    for t in range(6):
        frac = t / 5
        value = y_min + frac * (y_max - y_min)
        y = y_axis + frac * plot_h
        c.line(x_axis, y, x_axis + plot_w, y)
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.HexColor(PALETTE["text"]))
        label = f"{value:.1f}" if y_min else f"{int(value):,}"
        c.drawRightString(x_axis - 8, y - 3, label)

    c.setStrokeColor(colors.HexColor(PALETTE["text"]))
    c.setLineWidth(1)
    c.line(x_axis, y_axis, x_axis, y_axis + plot_h)
    c.line(x_axis, y_axis, x_axis + plot_w, y_axis)

    bar_slot = plot_w / len(labels)
    bar_w = bar_slot * 0.56
    for idx, (label, value) in enumerate(zip(labels, values)):
        x0 = x_axis + idx * bar_slot + (bar_slot - bar_w) / 2
        bar_h = (value - y_min) / (y_max - y_min) * plot_h
        c.setFillColor(colors.HexColor(colors_list[idx % len(colors_list)]))
        c.rect(x0, y_axis, bar_w, bar_h, fill=1, stroke=0)
        c.setFillColor(colors.HexColor(PALETTE["text"]))
        c.setFont("Helvetica-Bold", 10)
        c.drawCentredString(x0 + bar_w / 2, y_axis + bar_h + 8, value_fmt.format(value))
        c.setFont("Helvetica", 10)
        c.drawCentredString(x0 + bar_w / 2, y_axis - 18, label)

    c.setFont("Helvetica", 11)
    c.drawCentredString(x_axis + plot_w / 2, 0.22 * inch, ylabel)
    c.showPage()
    c.save()


def save_weight_png():
    width, height = 900, 500
    image = Image.new("RGB", (width, height), hex_to_rgb(PALETTE["background"]))
    draw = ImageDraw.Draw(image)
    title_font = load_font(32, bold=True)
    label_font = load_font(22)
    value_font = load_font(24, bold=True)
    text_color = hex_to_rgb(PALETTE["text"])
    colors_list = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"]]

    draw.text((70, 35), "Ensemble Weights", font=title_font, fill=text_color)
    x0, y0 = 90, 190
    total_w, h = 720, 76
    cursor = x0
    for idx, (label, weight) in enumerate(ENSEMBLE_WEIGHTS):
        w = total_w * weight
        color = hex_to_rgb(colors_list[idx])
        draw.rounded_rectangle((cursor, y0, cursor + w, y0 + h), radius=10, fill=color)
        draw_centered(draw, (cursor + w / 2, y0 + h / 2), f"{weight:.2f}", value_font, (255, 255, 255))
        cursor += w

    legend_x = 130
    for idx, (label, weight) in enumerate(ENSEMBLE_WEIGHTS):
        y = 330 + idx * 42
        draw.rounded_rectangle((legend_x, y, legend_x + 28, y + 28), radius=4, fill=hex_to_rgb(colors_list[idx]))
        draw.text((legend_x + 42, y - 1), f"{label}: {weight:.2f}", font=label_font, fill=text_color)

    image.save(os.path.join(FIGURE_DIR, "ensemble_weights.png"), dpi=(160, 160))


def save_weight_pdf():
    width, height = landscape((7.6 * inch, 4.2 * inch))
    path = os.path.join(FIGURE_DIR, "ensemble_weights.pdf")
    c = canvas.Canvas(path, pagesize=(width, height))
    c.setTitle("Ensemble Weights")
    colors_list = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"]]

    c.setFillColor(colors.HexColor(PALETTE["text"]))
    c.setFont("Helvetica-Bold", 18)
    c.drawString(0.65 * inch, height - 0.45 * inch, "Ensemble Weights")

    x0, y0 = 0.8 * inch, 2.15 * inch
    total_w, h = 6.0 * inch, 0.55 * inch
    cursor = x0
    for idx, (_, weight) in enumerate(ENSEMBLE_WEIGHTS):
        w = total_w * weight
        c.setFillColor(colors.HexColor(colors_list[idx]))
        c.rect(cursor, y0, w, h, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 12)
        c.drawCentredString(cursor + w / 2, y0 + h / 2 - 4, f"{weight:.2f}")
        cursor += w

    c.setFillColor(colors.HexColor(PALETTE["text"]))
    c.setFont("Helvetica", 12)
    for idx, (label, weight) in enumerate(ENSEMBLE_WEIGHTS):
        y = 1.35 * inch - idx * 0.32 * inch
        c.setFillColor(colors.HexColor(colors_list[idx]))
        c.rect(1.1 * inch, y, 0.16 * inch, 0.16 * inch, fill=1, stroke=0)
        c.setFillColor(colors.HexColor(PALETTE["text"]))
        c.drawString(1.35 * inch, y - 1, f"{label}: {weight:.2f}")

    c.showPage()
    c.save()


def rating_distribution():
    ratings = load_ratings("train.txt", has_score=True)
    counter = Counter(r for _, _, r in ratings)
    labels = [str(score) for score in range(10, 101, 10)]
    values = [counter[score] for score in range(10, 101, 10)]
    colors_list = [
        PALETTE["blue"],
        PALETTE["orange"],
        PALETTE["green"],
        PALETTE["red"],
        PALETTE["purple"],
    ]
    save_bar_png(
        "rating_distribution",
        "Training Rating Distribution",
        labels,
        values,
        "Rating score",
        colors_list,
        value_fmt="{:,.0f}",
        y_min=0,
    )
    save_bar_pdf(
        "rating_distribution",
        "Training Rating Distribution",
        labels,
        values,
        "Rating score",
        colors_list,
        value_fmt="{:,.0f}",
        y_min=0,
    )


def model_rmse_comparison():
    labels = [name for name, _ in MODEL_RMSE]
    values = [rmse for _, rmse in MODEL_RMSE]
    colors_list = [
        PALETTE["gray"],
        PALETTE["blue"],
        PALETTE["orange"],
        PALETTE["red"],
        PALETTE["green"],
    ]
    save_bar_png(
        "model_rmse_comparison",
        "Validation RMSE Comparison",
        labels,
        values,
        "Model",
        colors_list,
        value_fmt="{:.4f}",
        y_min=16.0,
    )
    save_bar_pdf(
        "model_rmse_comparison",
        "Validation RMSE Comparison",
        labels,
        values,
        "Model",
        colors_list,
        value_fmt="{:.4f}",
        y_min=16.0,
    )


def ensemble_weights():
    save_weight_png()
    save_weight_pdf()


def main():
    ensure_figure_dir()
    rating_distribution()
    model_rmse_comparison()
    ensemble_weights()
    print(f"Figures written to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
