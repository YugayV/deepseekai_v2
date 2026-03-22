"""
EURUSD AI Trading Bot - Railway Optimized
"""

import os
import json
import time
import logging
import pandas as pd
import numpy as np
import yfinance as yf
import talib
import joblib
from datetime import datetime
import openai

# Railway specific
RAILWAY = os.getenv("RAILWAY", False)
PORT = int(os.getenv("PORT", 8000))

# Настройка логирования для Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Для Railway health check
if RAILWAY:
    import threading
    from http.server import HTTPServer, BaseHTTPRequestHandler
    
    class HealthCheckHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/health':
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'OK')
            else:
                self.send_response(404)
                self.end_headers()
    
    def start_health_server():
        server = HTTPServer(('0.0.0.0', PORT), HealthCheckHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        logger.info(f"Health check server running on port {PORT}")
    
    start_health_server()

# Остальной код бота...