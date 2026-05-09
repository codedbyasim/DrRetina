#!/usr/bin/env python3
"""
RetinAgent — Entry Point
AMD Developer Hackathon 2026
"""

from ui import build_ui

demo = build_ui()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        ssr_mode=False,
    )
