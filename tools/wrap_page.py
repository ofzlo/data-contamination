#!/usr/bin/env python
"""Wrap tools/page.html (body fragment) into a standalone index.html for GitHub Pages."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
body = (root / "tools/page.html").read_text()
head = (
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
    "<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}"
    "body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n"
)
# move <title>/<meta>/<link>/<style> into <head>; the rest goes into <body>
split = body.index("</style>") + len("</style>")
(root / "index.html").write_text(head + body[:split] + "\n</head>\n<body>\n" + body[split:] + "\n</body>\n</html>\n")
