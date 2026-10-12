
from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from voice_assistant.intent.parser import ParsedIntent


THEME = Theme({
    "title": "bold cyan",
    "success": "bold green",
    "error": "bold red",
    "warning": "bold yellow",
    "muted": "dim",
    "label": "bold bright_white",
})


class UIFeedback:
    def __init__(self, hotkey_label: str = "Left Ctrl") -> None:
        self.console = Console(theme=THEME)
        self._hotkey_label = hotkey_label

    def header(self) -> None:
        self.console.clear()

        banner = Text()
        banner.append("  VOICE ", style="bold cyan")
        banner.append("FILE ASSISTANT", style="bold bright_white")
        banner.append("\n  Context-aware voice-controlled file management",
                      style="dim")

        self.console.print()
        self.console.print(
            Panel(
                banner,
                border_style="cyan",
                box=box.DOUBLE,
                padding=(1, 2),
            )
        )

        status = (
            f"[success]● Ready[/success]  "
            f"[muted]Hotkey:[/muted] [bold]{self._hotkey_label}[/bold]  "
            "[muted]Exit: Ctrl+C[/muted]"
        )
        self.console.print(f"  {status}\n")

    def show_current_dir(self, path: Path) -> None:
        self.console.print(
            Panel(
                f"[bold]📁 Current directory[/bold]\n[cyan]{path}[/cyan]",
                border_style="blue",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )

    def show_speech(self, text: str) -> None:
        self.console.print(
            Panel(
                f'[italic bright_white]"{text}"[/italic bright_white]',
                title="[bold cyan]🎙 Recognized speech[/bold cyan]",
                border_style="cyan",
                box=box.ROUNDED,
                padding=(0, 2),
            )
        )

    def show_intent(self, intent: ParsedIntent) -> None:
        intent_name = intent.intent.name.replace("_", " ").title()
        method = getattr(intent, "parse_method", "regex")
        confidence = getattr(intent, "confidence", 0.0)

        table = Table(
            show_header=False,
            box=box.SIMPLE,
            padding=(0, 1),
            expand=True,
        )
        table.add_column("Property", style="bold bright_white",
                         width=16, no_wrap=True)
        table.add_column("Value", style="white")

        table.add_row("Action", f"[cyan]{intent_name}[/cyan]")
        table.add_row("Confidence", f"{confidence:.0%}")
        table.add_row("Parse method", str(method))

        entities = [
            ("File", getattr(intent, "filename", None)),
            ("Source", getattr(intent, "source", None)),
            ("Destination", getattr(intent, "destination", None)),
            ("File type", getattr(intent, "file_type", None)),
            ("Folder", getattr(intent, "folder", None)),
            ("Date", getattr(intent, "date_expr", None)),
            ("Keywords", getattr(intent, "keywords", None)),
        ]

        for label, value in entities:
            if value:
                table.add_row(label, str(value))

        confidence_style = (
            "green" if confidence >= 0.8
            else "yellow" if confidence >= 0.5
            else "red"
        )

        self.console.print(
            Panel(
                table,
                title="[bold]🧠 Intent analysis[/bold]",
                subtitle=f"[{confidence_style}]Confidence: {confidence:.0%}[/{confidence_style}]",
                border_style=confidence_style,
                box=box.ROUNDED,
            )
        )

    def show_result(self, success: bool, message: str) -> None:
        if success:
            self.console.print(
                Panel(
                    f"[success]✓ {message}[/success]",
                    title="[bold green]SUCCESS[/bold green]",
                    border_style="green",
                    box=box.ROUNDED,
                )
            )
        else:
            self.console.print(
                Panel(
                    f"[error]✗ {message}[/error]",
                    title="[bold red]FAILED[/bold red]",
                    border_style="red",
                    box=box.ROUNDED,
                )
            )

    def show_error(self, message: str) -> None:
        self.console.print(
            Panel(
                f"[warning]⚠ {message}[/warning]",
                title="[bold yellow]NOTICE[/bold yellow]",
                border_style="yellow",
                box=box.ROUNDED,
            )
        )

    def show_search_results(self, results: list, query: str) -> None:
        if not results:
            self.console.print(
                Panel(
                    f'No files found for "{query}".',
                    title="[yellow]Search results[/yellow]",
                    border_style="yellow",
                )
            )
            return

        table = Table(
            title=f'Search results for "{query}"',
            box=box.ROUNDED,
            border_style="cyan",
            header_style="bold cyan",
            show_lines=True,
        )
        table.add_column("#", justify="right", style="dim", width=3)
        table.add_column("File", style="bold white")
        table.add_column("Match", style="yellow")
        table.add_column("Location", style="dim", overflow="fold")

        for i, result in enumerate(results, 1):
            match_type = getattr(result, "match_type", "") or "—"
            table.add_row(
                str(i),
                str(result.name),
                str(match_type),
                str(result.path.parent),
            )

        self.console.print(table)
        self.console.print(
            f"[success]Found {len(results)} result(s).[/success]\n"
        )

    def show_list_results(self, items: list[Path]) -> None:
        if not items:
            self.console.print("[muted]This directory is empty.[/muted]")
            return

        table = Table(
            title="Directory contents",
            box=box.ROUNDED,
            border_style="blue",
            header_style="bold cyan",
            show_lines=True,
        )
        table.add_column("Type", width=10)
        table.add_column("Name", style="white")
        table.add_column("Size", justify="right", style="dim")

        for path in items:
            try:
                if path.is_dir():
                    table.add_row("📁 Folder", path.name + "/", "—")
                elif path.is_file():
                    size = path.stat().st_size
                    table.add_row(
                        "📄 File",
                        path.name,
                        self._format_size(size),
                    )
            except OSError:
                table.add_row("?", path.name, "Unavailable")

        self.console.print(table)

    def _format_size(self, size: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
