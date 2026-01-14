#!/usr/bin/env python3
"""
HTTP Server for Render - Keeps service alive
Simple HTTP server that listens on PORT while background processes run
"""

import asyncio
import sys
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


class HealthCheckHandler(SimpleHTTPRequestHandler):
    """Simple HTTP handler for health checks"""

    def do_GET(self):
        # Respond with 200 OK for health checks
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b"KOL Tracker ML - All Systems Operational\n")

    def log_message(self, format, *args):
        # Suppress log messages
        pass


def start_http_server(port):
    """Start HTTP server in background thread"""
    try:
        with socketserver.TCPServer(("", port), HealthCheckHandler) as httpd:
            print(f"[*] HTTP server listening on port {port}")
            httpd.serve_forever()
    except Exception as e:
        print(f"[!] HTTP server error: {e}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8501))
    start_http_server(port)
