"""
Custom Python HTTP Server for ChurnVision AI Platform
Serves static web UI assets and provides light-weight REST API endpoints for model predictions,
dataset stats, model training, and file downloads.
"""

import http.server
import socketserver
import json
import urllib.parse
import os
import sys
from predict_churn import calculate_churn_risk

PORT = 8000

class ChurnServerHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        
        if path == '/api/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            status_data = {
                "status": "online",
                "python_version": sys.version,
                "dataset_exists": os.path.exists('telecom_churn.csv'),
                "model_results_exists": os.path.exists('model_results.json')
            }
            self.wfile.write(json.dumps(status_data).encode('utf-8'))
            return

        if path == '/api/results':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            if os.path.exists('model_results.json'):
                with open('model_results.json', 'r') as f:
                    content = f.read()
                self.wfile.write(content.encode('utf-8'))
            else:
                self.wfile.write(json.dumps({"error": "Model results not generated yet."}).encode('utf-8'))
            return

        super().do_GET()

    def do_POST(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        
        if path == '/api/predict':
            try:
                customer_data = json.loads(body)
                result = calculate_churn_risk(customer_data)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
            return
            
        if path == '/api/retrain':
            try:
                from train_model import run_training_pipeline
                summary = run_training_pipeline()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(summary).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
            return
            
        self.send_response(404)
        self.end_headers()

def run_server():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    handler = ChurnServerHandler
    ports_to_try = [8085, 8080, 8000, 8888, 5000]
    httpd = None
    selected_port = None
    for port in ports_to_try:
        try:
            httpd = socketserver.TCPServer(("", port), handler)
            selected_port = port
            break
        except OSError:
            continue
            
    if not httpd:
        print("Could not bind to any port.")
        return

    print(f"[INFO] ChurnVision AI Server running at http://localhost:{selected_port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")

if __name__ == '__main__':
    run_server()
