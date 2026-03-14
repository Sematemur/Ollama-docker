from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1600
HEIGHT = 980
BACKGROUND = "#f4f7fb"
TEXT = "#142235"
MUTED = "#5b6b7f"
LINE = "#7c8ea3"
SUCCESS = "#4caf7d"
WARNING = "#d8a33d"
ERROR = "#d96b7c"

NODES = {
    "START": {"xy": (110, 120), "size": (150, 68), "fill": "#17324d", "stroke": "#9cc8f2", "lines": ["START"]},
    "reasoning": {
        "xy": (350, 120),
        "size": (220, 88),
        "fill": "#1d5a58",
        "stroke": "#9fe0dd",
        "lines": ["reasoning", "Analyze the user query"],
    },
    "tool_selection": {
        "xy": (665, 120),
        "size": (250, 96),
        "fill": "#6a4f1f",
        "stroke": "#efd39a",
        "lines": ["tool_selection", "Choose tavily_search", "or no tool"],
    },
    "tool_execution": {
        "xy": (1080, 230),
        "size": (245, 96),
        "fill": "#4b3c79",
        "stroke": "#cbb9ff",
        "lines": ["tool_execution", "Run selected tool"],
    },
    "response_generation": {
        "xy": (665, 390),
        "size": (275, 96),
        "fill": "#29506f",
        "stroke": "#a9d2f4",
        "lines": ["response_generation", "Produce JSON answer"],
    },
    "validation": {
        "xy": (665, 580),
        "size": (220, 96),
        "fill": "#72591f",
        "stroke": "#efd89f",
        "lines": ["validation", "Check schema", "and retry policy"],
    },
    "fallback": {
        "xy": (1040, 580),
        "size": (190, 96),
        "fill": "#7a3243",
        "stroke": "#f0b0bf",
        "lines": ["fallback", "Safe response"],
    },
    "END": {"xy": (1330, 580), "size": (140, 68), "fill": "#17324d", "stroke": "#9cc8f2", "lines": ["END"]},
}

EDGES = [
    ("START", "reasoning", "", LINE),
    ("reasoning", "tool_selection", "", LINE),
    ("tool_selection", "tool_execution", "selected_tool", SUCCESS),
    ("tool_selection", "response_generation", "no tool", WARNING),
    ("tool_execution", "response_generation", "tool_result", LINE),
    ("response_generation", "validation", "", LINE),
    ("validation", "END", "validation_passed", SUCCESS),
    ("validation", "tool_execution", "needs_forced_tool", WARNING),
    ("validation", "response_generation", "needs_regeneration", WARNING),
    ("validation", "fallback", "retry_count > max_retries", ERROR),
    ("fallback", "END", "", ERROR),
]

STATE_LINES = [
    "GraphState carries: user_message, history, available_tools, tool_decision, selected_tool,",
    "tool_result, retry_count, validation_report, response_envelope, final_response, fallback_report",
]


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = ["arialbd.ttf", "segoeuib.ttf"] if bold else ["arial.ttf", "segoeui.ttf"]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


TITLE_FONT = load_font(34, bold=True)
SUBTITLE_FONT = load_font(18)
NODE_TITLE_FONT = load_font(20, bold=True)
NODE_TEXT_FONT = load_font(15)
LABEL_FONT = load_font(15, bold=True)
STATE_FONT = load_font(17)


def center(node_name: str) -> tuple[int, int]:
    x, y = NODES[node_name]["xy"]
    w, h = NODES[node_name]["size"]
    return x + w // 2, y + h // 2


def draw_node(draw: ImageDraw.ImageDraw, name: str) -> None:
    node = NODES[name]
    x, y = node["xy"]
    w, h = node["size"]
    radius = 24
    shadow_offset = 8

    draw.rounded_rectangle(
        [x + shadow_offset, y + shadow_offset, x + w + shadow_offset, y + h + shadow_offset],
        radius=radius,
        fill="#d9e3ef",
    )
    draw.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=node["fill"], outline=node["stroke"], width=3)

    lines = node["lines"]
    current_y = y + 22
    for index, line in enumerate(lines):
        font = NODE_TITLE_FONT if index == 0 else NODE_TEXT_FONT
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        draw.text((x + (w - text_width) / 2, current_y), line, font=font, fill="white")
        current_y += 26 if index == 0 else 22


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str) -> None:
    draw.line([start, end], fill=color, width=5)
    arrow_size = 10
    ex, ey = end
    if abs(end[0] - start[0]) > abs(end[1] - start[1]):
        points = [(ex, ey), (ex - arrow_size, ey - arrow_size // 2), (ex - arrow_size, ey + arrow_size // 2)]
    else:
        points = [(ex, ey), (ex - arrow_size // 2, ey - arrow_size), (ex + arrow_size // 2, ey - arrow_size)]
    draw.polygon(points, fill=color)


def draw_edge(draw: ImageDraw.ImageDraw, source: str, target: str, label: str, color: str) -> None:
    sx, sy = center(source)
    tx, ty = center(target)

    if source == "tool_selection" and target == "tool_execution":
        start = (NODES[source]["xy"][0] + NODES[source]["size"][0], sy)
        mid1 = (start[0] + 100, start[1])
        mid2 = (tx, NODES[target]["xy"][1] - 35)
        end = (tx, NODES[target]["xy"][1])
        draw.line([start, mid1, mid2, end], fill=color, width=5)
        draw_arrow(draw, mid2, end, color)
        label_xy = (start[0] + 60, start[1] - 34)
    elif source == "tool_selection" and target == "response_generation":
        start = (sx, NODES[source]["xy"][1] + NODES[source]["size"][1])
        end = (sx, NODES[target]["xy"][1])
        draw_arrow(draw, start, end, color)
        label_xy = (start[0] + 16, (start[1] + end[1]) // 2 - 14)
    elif source == "validation" and target == "tool_execution":
        start = (NODES[source]["xy"][0] + NODES[source]["size"][0], sy)
        mid1 = (start[0] + 120, start[1])
        mid2 = (tx, NODES[target]["xy"][1] + NODES[target]["size"][1] + 35)
        end = (tx, NODES[target]["xy"][1] + NODES[target]["size"][1])
        draw.line([start, mid1, mid2, end], fill=color, width=5)
        draw_arrow(draw, mid2, end, color)
        label_xy = (start[0] + 90, start[1] - 34)
    elif source == "validation" and target == "response_generation":
        start = (sx, NODES[source]["xy"][1])
        end = (sx, NODES[target]["xy"][1] + NODES[target]["size"][1])
        draw_arrow(draw, start, end, color)
        label_xy = (start[0] + 16, (start[1] + end[1]) // 2 - 14)
    else:
        if abs(tx - sx) >= abs(ty - sy):
            start = (NODES[source]["xy"][0] + NODES[source]["size"][0], sy)
            end = (NODES[target]["xy"][0], ty)
        else:
            start = (sx, NODES[source]["xy"][1] + NODES[source]["size"][1])
            end = (tx, NODES[target]["xy"][1])
        draw_arrow(draw, start, end, color)
        label_xy = ((start[0] + end[0]) // 2 + 10, (start[1] + end[1]) // 2 - 22)

    if label:
        bbox = draw.textbbox((0, 0), label, font=LABEL_FONT)
        pad = 10
        draw.rounded_rectangle(
            [label_xy[0] - pad, label_xy[1] - pad, label_xy[0] + (bbox[2] - bbox[0]) + pad, label_xy[1] + (bbox[3] - bbox[1]) + pad],
            radius=14,
            fill="white",
            outline=color,
            width=2,
        )
        draw.text(label_xy, label, font=LABEL_FONT, fill=color)


def draw_state_panel(draw: ImageDraw.ImageDraw) -> None:
    x, y, w, h = 80, 790, 1440, 120
    draw.rounded_rectangle([x, y, x + w, y + h], radius=24, fill="#ffffff", outline="#c8d6e5", width=2)
    draw.text((105, 820), "GraphState", font=NODE_TITLE_FONT, fill=TEXT)
    for idx, line in enumerate(STATE_LINES):
        draw.text((105, 855 + idx * 26), line, font=STATE_FONT, fill=MUTED)


def main() -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    draw.text((80, 52), "LangGraph Orchestrator", font=TITLE_FONT, fill=TEXT)
    draw.text(
        (80, 98),
        "Simple execution flow: select tool, generate answer, validate, retry if needed.",
        font=SUBTITLE_FONT,
        fill=MUTED,
    )

    for edge in EDGES:
        draw_edge(draw, *edge)

    for name in NODES:
        draw_node(draw, name)

    draw_state_panel(draw)

    output_path = Path(__file__).with_name("orchestrator_graph.png")
    image.save(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
