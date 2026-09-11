from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pythoncom
import win32com.client


ROOT = Path(__file__).resolve().parent
SPEC_PATH = ROOT / "figure_spec.json"

COLORS = {
    "ink": "#111111",
    "line": "#111111",
    "panel": "#F7F7F7",
    "light": "#F1F1F1",
    "green": "#CDE8C5",
    "green_dark": "#2C9B3F",
    "red": "#B40000",
    "white": "#FFFFFF",
}


def rgb(value: str) -> str:
    value = value.strip("#")
    return f"RGB({int(value[0:2], 16)},{int(value[2:4], 16)},{int(value[4:6], 16)})"


class Canvas:
    def __init__(self, app: Any, width: float, height: float) -> None:
        self.doc = app.Documents.Add("")
        self.page = app.ActivePage
        self.width = width
        self.height = height
        self.page.PageSheet.CellsU("PageWidth").FormulaU = f"{width} in"
        self.page.PageSheet.CellsU("PageHeight").FormulaU = f"{height} in"
        self.font_id = self._font_id("Microsoft YaHei") or self._font_id("Arial")
        self.shapes: dict[str, Any] = {}

    def _font_id(self, name: str) -> int | None:
        try:
            return self.doc.Fonts.Add(name).ID
        except Exception:
            for idx in range(1, self.doc.Fonts.Count + 1):
                font = self.doc.Fonts.Item(idx)
                if name.lower() in font.Name.lower():
                    return font.ID
        return None

    def y(self, top: float) -> float:
        return self.height - top

    def apply_text(self, shape: Any, size: float, bold: bool = False, color: str = COLORS["ink"]) -> None:
        shape.CellsU("Char.Size").FormulaU = f"{size} pt"
        shape.CellsU("Char.Color").FormulaU = rgb(color)
        shape.CellsU("Char.Style").FormulaU = "1" if bold else "0"
        shape.CellsU("Para.HorzAlign").FormulaU = "1"
        shape.CellsU("VerticalAlign").FormulaU = "1"
        if self.font_id is not None:
            shape.CellsU("Char.Font").FormulaU = str(self.font_id)
        for margin in ("LeftMargin", "RightMargin", "TopMargin", "BottomMargin"):
            shape.CellsU(margin).FormulaU = "0.035 in"

    def rect(
        self,
        name: str,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str = "",
        fill: str = COLORS["white"],
        line: str = COLORS["line"],
        size: float = 10,
        bold: bool = False,
        radius: float = 0.06,
        line_w: float = 1.1,
        font: str = COLORS["ink"],
        angle: float = 0,
    ) -> Any:
        shape = self.page.DrawRectangle(x, self.y(y + h), x + w, self.y(y))
        shape.NameU = name
        shape.Text = text
        shape.CellsU("FillForegnd").FormulaU = rgb(fill)
        shape.CellsU("LineColor").FormulaU = rgb(line)
        shape.CellsU("LineWeight").FormulaU = f"{line_w} pt"
        shape.CellsU("Rounding").FormulaU = f"{radius} in"
        if angle:
            shape.CellsU("Angle").FormulaU = f"{angle} deg"
        self.apply_text(shape, size=size, bold=bold, color=font)
        self.shapes[name] = shape
        return shape

    def oval(
        self,
        name: str,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str = "",
        fill: str = COLORS["white"],
        line: str = COLORS["line"],
        size: float = 10,
        bold: bool = False,
        line_w: float = 1.0,
    ) -> Any:
        shape = self.page.DrawOval(x, self.y(y + h), x + w, self.y(y))
        shape.NameU = name
        shape.Text = text
        shape.CellsU("FillForegnd").FormulaU = rgb(fill)
        shape.CellsU("LineColor").FormulaU = rgb(line)
        shape.CellsU("LineWeight").FormulaU = f"{line_w} pt"
        self.apply_text(shape, size=size, bold=bold)
        self.shapes[name] = shape
        return shape

    def text(
        self,
        name: str,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        size: float = 10,
        bold: bool = False,
        color: str = COLORS["ink"],
        align: int = 1,
        angle: float = 0,
    ) -> Any:
        shape = self.rect(name, x, y, w, h, text, fill=COLORS["white"], line=COLORS["white"],
                          size=size, bold=bold, radius=0, line_w=0, font=color, angle=angle)
        shape.CellsU("FillPattern").FormulaU = "0"
        shape.CellsU("LinePattern").FormulaU = "0"
        shape.CellsU("Para.HorzAlign").FormulaU = str(align)
        return shape

    def vertical_text(
        self,
        name: str,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        size: float = 10,
        bold: bool = False,
    ) -> Any:
        shape = self.text(name, x, y, w, h, text, size=size, bold=bold)
        shape.CellsU("TxtAngle").FormulaU = "90 deg"
        return shape

    def line(
        self,
        name: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        arrow: bool = True,
        color: str = COLORS["line"],
        weight: float = 1.2,
        dashed: bool = False,
    ) -> Any:
        shape = self.page.DrawLine(x1, self.y(y1), x2, self.y(y2))
        shape.NameU = name
        shape.CellsU("LineColor").FormulaU = rgb(color)
        shape.CellsU("LineWeight").FormulaU = f"{weight} pt"
        if arrow:
            shape.CellsU("EndArrow").FormulaU = "4"
        if dashed:
            shape.CellsU("LinePattern").FormulaU = "2"
        return shape

    def polyline(
        self,
        prefix: str,
        points: list[tuple[float, float]],
        arrow: bool = True,
        color: str = COLORS["line"],
        weight: float = 1.2,
        dashed: bool = False,
    ) -> None:
        for idx in range(len(points) - 1):
            self.line(f"{prefix}_{idx}", *points[idx], *points[idx + 1],
                      arrow=arrow and idx == len(points) - 2, color=color,
                      weight=weight, dashed=dashed)


def draw_cylinder(c: Canvas, name: str, x: float, y: float, w: float, h: float, text: str, size: float = 10) -> None:
    c.rect(f"{name}_body", x, y + 0.14, w, h - 0.28, "", fill=COLORS["light"], line=COLORS["line"],
           radius=0, line_w=1.0)
    c.oval(f"{name}_top", x, y, w, 0.28, "", fill=COLORS["light"], line=COLORS["line"], line_w=1.0)
    c.oval(f"{name}_bottom", x, y + h - 0.28, w, 0.28, "", fill=COLORS["light"], line=COLORS["line"], line_w=1.0)
    c.text(f"{name}_label", x + 0.06, y + 0.17, w - 0.12, h - 0.28, text, size=size)
    c.shapes[name] = c.shapes[f"{name}_body"]


def draw_client(c: Canvas, idx: str, x: float, y: float) -> None:
    c.rect(f"client_{idx}_group", x, y, 1.92, 2.04, "", fill=COLORS["white"], line=COLORS["line"],
           radius=0.10, line_w=0.9)
    c.shapes[f"client_{idx}_group"].CellsU("LinePattern").FormulaU = "2"
    c.text(f"client_{idx}_title", x + 0.10, y + 0.04, 1.72, 0.24, f"Client {idx} client", size=9.8, bold=True)
    c.rect(f"client_{idx}_llm", x + 0.16, y + 0.34, 1.60, 0.34, "Local Legal LLM",
           fill=COLORS["light"], line=COLORS["line"], size=9.0, radius=0.07, line_w=0.9)
    c.rect(f"client_{idx}_retrieval", x + 0.16, y + 0.86, 1.60, 0.32, "Private Retrieval",
           fill=COLORS["light"], line=COLORS["line"], size=8.9, radius=0.07, line_w=0.9)
    c.rect(f"client_{idx}_memory", x + 0.22, y + 1.43, 1.48, 0.54,
           f"Jurisdiction\nMemory M^jur_{idx}", fill=COLORS["light"], line=COLORS["line"],
           size=8.8, radius=0.07, line_w=0.9)
    c.line(f"client_{idx}_up", x + 0.96, y + 0.86, x + 0.96, y + 0.68, color=COLORS["line"], weight=1.0)
    c.line(f"client_{idx}_down", x + 0.96, y + 1.18, x + 0.96, y + 1.43, color=COLORS["line"], weight=1.0)


def draw_swimlanes(c: Canvas) -> None:
    c.rect("outer_frame", 0.10, 0.38, 13.25, 7.12, "", fill=COLORS["white"], line=COLORS["line"],
           radius=0, line_w=1.0)
    c.line("lane_divider_1", 0.10, 2.65, 13.35, 2.65, arrow=False, color=COLORS["line"], weight=1.0)
    c.line("lane_divider_2", 0.10, 5.45, 13.35, 5.45, arrow=False, color=COLORS["line"], weight=1.0)
    c.line("label_divider", 0.72, 0.38, 0.72, 7.50, arrow=False, color=COLORS["line"], weight=1.0)
    c.vertical_text("lane_label_1", 0.16, 0.62, 0.48, 1.85,
                    "Query and Shared Deliberation State", size=9.8, bold=True)
    c.vertical_text("lane_label_2", 0.16, 2.92, 0.48, 2.20,
                    "Private Institutional Reasoning", size=9.8, bold=True)
    c.vertical_text("lane_label_3", 0.16, 5.70, 0.48, 1.62,
                    "Verification, Decision, and Audit", size=9.3, bold=True)


def draw_top_lane(c: Canvas) -> None:
    c.text("legal_query", 0.96, 0.88, 1.70, 0.42, "Legal Query q +\nTarget Jurisdiction j", size=10.2)
    c.line("query_to_parse", 2.55, 1.18, 2.86, 1.18, color=COLORS["line"], weight=1.1)
    c.rect("parse", 2.86, 0.78, 1.72, 0.78, "", fill=COLORS["light"], line=COLORS["line"],
           radius=0.10, line_w=1.0)
    c.text("parse_title", 3.10, 0.88, 1.22, 0.22, "Parse", size=11.4, bold=True)
    c.text("parse_body", 3.02, 1.14, 1.40, 0.34, "facts, issues,\nevidence descriptors", size=8.9)
    c.line("parse_to_align", 4.58, 1.18, 5.08, 1.18, color=COLORS["line"], weight=1.1)
    c.rect("align", 5.08, 0.78, 1.92, 0.78, "", fill=COLORS["light"], line=COLORS["line"],
           radius=0.10, line_w=1.0)
    c.text("align_title", 5.45, 0.88, 1.18, 0.22, "Align", size=11.4, bold=True)
    c.text("align_body", 5.20, 1.14, 1.68, 0.34, "jurisdiction, authority\nhierarchy, rule mappings", size=8.7)
    c.text("typed_message", 7.42, 0.90, 1.16, 0.24, "Typed Message", size=9.4)
    c.line("align_to_prosecutor", 7.00, 1.18, 8.92, 1.18, color=COLORS["line"], weight=1.1)

    c.rect("prosecutor", 8.92, 0.78, 1.36, 0.78, "Prosecutor\nAgent",
           fill=COLORS["light"], line=COLORS["line"], size=10.8, bold=True, radius=0.10, line_w=1.0)
    c.rect("defence", 11.42, 0.78, 1.44, 0.78, "Defence\nAgent",
           fill=COLORS["light"], line=COLORS["line"], size=10.8, bold=True, radius=0.10, line_w=1.0)
    c.line("pros_to_def", 10.28, 1.12, 11.42, 1.12, color=COLORS["line"], weight=1.0)
    c.line("def_to_pros", 11.42, 1.38, 10.28, 1.38, color=COLORS["line"], weight=1.0)
    c.text("read_from_top", 10.40, 0.88, 0.86, 0.22, "Read from", size=8.8)
    c.text("update_top", 10.48, 1.48, 0.70, 0.20, "Update", size=8.8)
    c.text("typed_sanitized", 8.58, 1.62, 1.32, 0.50, "Typed\nSanitized\nMessage m_k", size=8.8)

    c.rect("case_memory_box", 9.82, 2.02, 3.02, 0.58,
           "Case Memory M^case\ncontradiction flags", fill=COLORS["light"], line=COLORS["line"],
           size=9.4, radius=0.09, line_w=1.0)
    c.line("pros_to_case", 9.66, 1.56, 9.66, 2.02, color=COLORS["line"], weight=1.0)
    c.line("def_to_case", 12.10, 1.56, 12.10, 2.02, color=COLORS["line"], weight=1.0)


def draw_private_lane(c: Canvas) -> None:
    xs = [0.84, 2.90, 4.92, 7.64]
    ids = ["1", "2", "3", "N"]
    for idx, (xx, cid) in enumerate(zip(xs, ids)):
        draw_client(c, cid, xx, 2.87)
    c.rect("schema_box", 0.76, 4.92, 2.44, 0.48,
           "Exact compact schema:\nm_k = (C_k, rho_k, r_k, pi_k, privacy_tag_k)",
           fill=COLORS["white"], line=COLORS["line"], size=7.9, radius=0.08, line_w=0.9)
    c.text("admissible_note", 3.28, 5.00, 5.60, 0.28,
           "Raw cases and client-local retrieval content are inadmissible on inter-client edges",
           size=8.8, align=0)

    # Client routes to prosecutor, with sanitized-message path.
    c.polyline("client1_route", [(1.80, 2.87), (1.80, 2.24), (7.72, 2.24), (7.72, 1.30), (8.92, 1.30)],
               color=COLORS["line"], weight=1.0)
    c.polyline("client2_route", [(3.86, 2.87), (3.86, 2.36), (7.86, 2.36), (7.86, 1.42), (8.92, 1.42)],
               color=COLORS["line"], weight=1.0)
    c.polyline("client3_route", [(5.88, 2.87), (5.88, 2.48), (8.02, 2.48), (8.02, 1.54), (8.92, 1.54)],
               color=COLORS["red"], weight=1.0, dashed=True)
    c.polyline("clientn_route", [(8.60, 2.87), (8.60, 2.56), (9.56, 2.56), (9.56, 1.56)],
               color=COLORS["red"], weight=1.0, dashed=True)
    c.line("sanitized_up", 8.52, 3.10, 8.52, 1.56, color=COLORS["red"], weight=1.0, dashed=True)
    c.text("ellipsis_clients", 7.08, 3.54, 0.40, 0.30, "...", size=16, bold=True)

    draw_cylinder(c, "case_memory", 10.04, 3.58, 2.38, 0.92,
                  "Case Memory M^case\nissues, disclosed arguments,\ncontradiction flags", size=8.8)
    c.line("case_flags_1", 10.74, 2.60, 10.74, 3.58, color=COLORS["line"], weight=1.0)
    c.line("case_flags_2", 11.24, 2.60, 11.24, 3.58, color=COLORS["line"], weight=1.0)
    c.line("case_flags_3", 11.74, 2.60, 11.74, 3.58, color=COLORS["line"], weight=1.0)
    c.text("read_case", 9.90, 4.88, 0.56, 0.22, "Read", size=8.9)
    c.text("arguments_case", 10.88, 4.88, 0.82, 0.22, "Arguments", size=8.9)


def draw_bottom_lane(c: Canvas) -> None:
    draw_cylinder(c, "citation_memory", 0.82, 6.20, 1.52, 0.70,
                  "Citation Memory\nM^cit", size=9.0)
    c.text("citation_checks", 0.82, 6.94, 1.52, 0.48,
           "canonical identifiers,\nvalidity,\nauthority-level checks", size=8.4)
    c.line("citation_to_verifier", 2.34, 6.54, 2.58, 6.54, color=COLORS["line"], weight=1.0)
    c.rect("citation_verifier", 2.58, 5.86, 1.34, 1.42, "Citation\nVerifier",
           fill=COLORS["light"], line=COLORS["line"], size=10.3, bold=True, radius=0.08, line_w=1.0)
    c.line("verifier_to_detector", 3.92, 6.56, 4.28, 6.56, color=COLORS["line"], weight=1.0)
    c.rect("conflict_detector", 4.28, 5.86, 1.40, 1.42, "", fill=COLORS["light"], line=COLORS["line"],
           radius=0.08, line_w=1.0)
    c.text("detector_title", 4.40, 5.94, 1.16, 0.40, "Conflict\nDetector", size=10.0, bold=True)
    for idx, (xx, yy, label) in enumerate([
        (4.42, 6.52, "D_cit"), (5.06, 6.52, "D_rea"),
        (4.42, 6.92, "D_ver"), (5.06, 6.92, "D_rule")
    ]):
        c.rect(f"detector_metric_{idx}", xx, yy, 0.48, 0.30, label,
               fill=COLORS["light"], line=COLORS["line"], size=7.5, radius=0.06, line_w=0.8)
    c.line("detector_to_auditor", 5.68, 6.56, 6.02, 6.56, color=COLORS["line"], weight=1.0)
    c.rect("privacy_auditor", 6.02, 5.86, 1.48, 1.42, "", fill=COLORS["light"], line=COLORS["line"],
           radius=0.08, line_w=1.0)
    c.text("auditor_title", 6.20, 5.94, 1.12, 0.40, "Privacy\nAuditor", size=10.2, bold=True)
    c.text("auditor_body", 6.16, 6.42, 1.20, 0.68,
           "sanitization\nprivacy_tag\npolicy compliance", size=8.4)
    c.line("auditor_to_gate", 7.50, 6.56, 7.90, 6.56, color=COLORS["line"], weight=1.0)

    # Diamond gate: rotated square with unrotated label.
    c.rect("verification_gate", 7.96, 5.94, 1.26, 1.26, "", fill=COLORS["green"], line=COLORS["line"],
           radius=0, line_w=1.0, angle=45)
    c.text("gate_label", 8.00, 6.05, 1.20, 0.98,
           "Citation\nverification\nPASS AND\nprivacy audit\nPASS?", size=8.6)
    c.line("gate_pass", 9.22, 6.56, 9.98, 6.56, color=COLORS["green_dark"], weight=1.5)
    c.text("pass_label", 9.44, 6.34, 0.38, 0.20, "PASS", size=9.0, color=COLORS["green_dark"])
    c.rect("judge", 9.98, 5.98, 1.62, 0.96,
           "Judge Agent\nverified arguments +\nverdict distributions\n+ conflict scores",
           fill=COLORS["light"], line=COLORS["line"], size=8.7, radius=0.08, line_w=1.0)
    c.polyline("gate_fail", [(8.58, 7.20), (8.58, 7.36), (9.98, 7.36), (9.98, 7.14)],
               color=COLORS["red"], weight=1.2, dashed=True)
    c.text("fail_label", 9.18, 7.18, 0.38, 0.20, "FAIL", size=9.0, color=COLORS["red"])
    c.rect("abstention", 9.98, 7.10, 1.62, 0.52, "Abstention",
           fill=COLORS["light"], line=COLORS["line"], size=10.0, bold=True, radius=0.08, line_w=1.0)
    c.line("judge_verdict", 11.60, 6.18, 11.94, 6.18, color=COLORS["line"], weight=1.0)
    c.line("judge_citations", 11.60, 6.56, 11.94, 6.56, color=COLORS["line"], weight=1.0)
    c.line("judge_audit", 11.60, 6.92, 11.94, 6.92, color=COLORS["line"], weight=1.0)
    c.text("verdict_output", 11.98, 6.04, 1.20, 0.24, "Verified Verdict y*", size=9.0, align=0)
    c.text("citation_output", 11.98, 6.42, 1.20, 0.24, "Valid Citations C*", size=9.0, align=0)
    c.text("audit_output", 11.98, 6.78, 1.00, 0.24, "Audit Record", size=9.0, align=0)
    c.line("abstention_out", 11.60, 7.36, 12.04, 7.36, color=COLORS["line"], weight=1.0)
    c.text("failed_annotations", 12.10, 7.18, 1.08, 0.34, "failed-check\nannotations", size=8.7, align=0)

    # Routes from private/schema and case memory into verification.
    c.line("schema_route", 3.20, 5.36, 3.20, 5.86, color=COLORS["line"], weight=1.0)
    c.text("route_label", 2.76, 5.52, 0.40, 0.20, "Route", size=8.6)
    c.polyline("detect_route", [(10.60, 4.50), (10.60, 5.58), (4.98, 5.58), (4.98, 5.86)],
               color=COLORS["line"], weight=1.0)
    c.text("detect_label", 4.18, 5.52, 0.52, 0.20, "Detects", size=8.6)
    c.polyline("privacy_checks", [(8.54, 4.90), (8.54, 5.70), (6.76, 5.70), (6.76, 5.86)],
               color=COLORS["red"], weight=1.0, dashed=True)
    c.text("checks_label", 5.92, 5.52, 0.46, 0.20, "checks", size=8.6)
    c.polyline("case_to_judge", [(10.34, 4.50), (10.34, 5.74), (10.78, 5.74), (10.78, 5.98)],
               color=COLORS["line"], weight=1.0)


def draw_trace_log(c: Canvas) -> None:
    draw_cylinder(c, "trace_log", 13.46, 2.78, 0.50, 2.02, "Reasoning trace log", size=9.2)
    c.shapes["trace_log_label"].CellsU("TxtAngle").FormulaU = "90 deg"
    c.polyline("def_to_log", [(12.86, 1.16), (13.72, 1.16), (13.72, 2.78)],
               color=COLORS["line"], weight=1.0)
    c.polyline("judge_to_log", [(13.18, 6.56), (13.72, 6.56), (13.72, 4.80)],
               color=COLORS["line"], weight=1.0)
    c.polyline("abstain_to_log", [(13.18, 7.36), (13.98, 7.36), (13.98, 4.80)],
               color=COLORS["line"], weight=1.0)
    c.polyline("log_to_prosecutor", [(13.98, 4.00), (14.04, 4.00), (14.04, 0.48), (9.60, 0.48), (9.60, 0.78)],
               color=COLORS["line"], weight=1.0)


def render() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    outputs = [ROOT / spec["exports"][key] for key in ("vsdx", "png", "emf")]
    for path in outputs:
        if path.exists():
            path.unlink()
    pythoncom.CoInitialize()
    app = win32com.client.DispatchEx("Visio.Application")
    app.Visible = False
    app.AlertResponse = 7
    canvas = Canvas(app, spec["canvas"]["width_in"], spec["canvas"]["height_in"])
    canvas.text("main_title", 3.50, 0.02, 7.10, 0.34,
                "Privacy-Preserving Multi-Agent Judicial Deliberation", size=16.8, bold=True)
    draw_swimlanes(canvas)
    draw_top_lane(canvas)
    draw_private_lane(canvas)
    draw_bottom_lane(canvas)
    draw_trace_log(canvas)
    canvas.page.Export(str(outputs[1]))
    canvas.page.Export(str(outputs[2]))
    time.sleep(0.2)
    canvas.doc.SaveAs(str(outputs[0]))
    canvas.doc.Close()
    app.Quit()
    pythoncom.CoUninitialize()


if __name__ == "__main__":
    render()
