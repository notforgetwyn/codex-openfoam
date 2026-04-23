from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ThemePalette:
    name: str
    window_bg: str
    surface_bg: str
    panel_bg: str
    sidebar_bg: str
    activity_bg: str
    status_bg: str
    border: str
    text: str
    muted_text: str
    accent: str


THEMES: dict[str, ThemePalette] = {
    "vscode-dark": ThemePalette(
        name="vscode-dark",
        window_bg="#181818",
        surface_bg="#1e1e1e",
        panel_bg="#181818",
        sidebar_bg="#252526",
        activity_bg="#333333",
        status_bg="#007acc",
        border="#2d2d30",
        text="#cccccc",
        muted_text="#9d9d9d",
        accent="#0e639c",
    ),
    "vscode-light": ThemePalette(
        name="vscode-light",
        window_bg="#f3f3f3",
        surface_bg="#ffffff",
        panel_bg="#f3f3f3",
        sidebar_bg="#ececec",
        activity_bg="#dcdcdc",
        status_bg="#007acc",
        border="#d4d4d4",
        text="#333333",
        muted_text="#616161",
        accent="#005fb8",
    ),
    "vscode-blue": ThemePalette(
        name="vscode-blue",
        window_bg="#11161c",
        surface_bg="#151b23",
        panel_bg="#10151c",
        sidebar_bg="#1b2530",
        activity_bg="#0f1720",
        status_bg="#1f6feb",
        border="#263241",
        text="#d7e2f0",
        muted_text="#8da1b9",
        accent="#3b82f6",
    ),
}


def build_stylesheet(theme_name: str, background_color: str | None = None) -> str:
    palette = THEMES.get(theme_name, THEMES["vscode-dark"])
    window_bg = background_color or palette.window_bg
    return f"""
    QMainWindow {{
        background-color: {window_bg};
        color: {palette.text};
    }}
    QWidget {{
        color: {palette.text};
        background-color: {palette.surface_bg};
    }}
    QMenuBar, QToolBar {{
        background-color: {palette.surface_bg};
        border-bottom: 1px solid {palette.border};
    }}
    QMenuBar::item:selected, QMenu::item:selected, QToolButton:hover {{
        background-color: {palette.accent};
    }}
    QStatusBar {{
        background-color: {palette.status_bg};
        color: #ffffff;
    }}
    QTabWidget::pane, QTreeWidget, QTextEdit, QListWidget, QFrame {{
        border: 1px solid {palette.border};
    }}
    QTabBar::tab {{
        background: {palette.surface_bg};
        color: {palette.muted_text};
        padding: 8px 14px;
        border: 1px solid {palette.border};
        border-bottom: none;
    }}
    QTabBar::tab:selected {{
        color: {palette.text};
        background: {palette.panel_bg};
    }}
    QTreeWidget, QListWidget {{
        background-color: {palette.sidebar_bg};
    }}
    QTextEdit {{
        background-color: {palette.panel_bg};
        selection-background-color: {palette.accent};
    }}
    """
