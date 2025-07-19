from http.server import BaseHTTPRequestHandler
import os
import sys
import subprocess
import streamlit as st

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/mount/src/ml-etf-rebalancer/')))

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        # In Vercel, we need to redirect to Streamlit Cloud
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML ETF Rebalancer - Redirecting</title>
            <meta http-equiv="refresh" content="0;URL='https://ml-etf-rebalancer.streamlit.app'" />
            <script>
                window.location.href = "https://ml-etf-rebalancer.streamlit.app";
            </script>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    text-align: center;
                }
            </style>
        </head>
        <body>
            <h1>ML ETF Rebalancer</h1>
            <p>Redirecting to Streamlit app...</p>
            <p>If you are not redirected automatically, <a href="https://ml-etf-rebalancer.streamlit.app">click here</a>.</p>
        </body>
        </html>
        """
        
        self.wfile.write(html.encode())
        return
