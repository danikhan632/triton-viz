import os
import sys
import time
import threading
import requests
import pandas as pd
import numpy as np
import torch
import triton
import triton.language as tl
from flask import Flask, render_template, jsonify, request, Response, send_from_directory
from flask_cors import CORS
from flask_cloudflared import _run_cloudflared

from .interpreter import get_data

from .tooltip import get_tooltip_data
from .trace import get_blocks, get_src




# Configure Flask application
# Assuming `frontend/build` contains your React build output.
app = Flask(
    __name__,
    static_folder='build',  
    static_url_path=''
)
CORS(app)

# Global variables to store data and state
global_data = None
raw_tensor_data = None
precomputed_c_values = {}
current_fullscreen_op = None





@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    """
    Serve the front-end React application. If a file is requested and it exists in the 
    build directory, serve it; otherwise, serve index.html.
    """
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/data')
def get_dat():
    """
    Return the global data as JSON. If data is not computed yet, update it first.
    """

    return jsonify(get_data())


@app.route('/api/setop', methods=['POST'])
def set_current_op():
    """
    Set the current fullscreen operation (op) by its UUID.
    """
    global current_fullscreen_op
    data = request.json
    current_fullscreen_op = data.get('uuid')
    return jsonify({"status": "Current op set successfully", "uuid": current_fullscreen_op})

@app.route('/get_src', methods=['GET'])
def getSrc():
    """
    Return the source code associated with the kernel or operations.
    """
    return Response(get_src(), mimetype='text/plain')






@app.route('/process_blocks', methods=['POST'])
def process_blocks():
    data = request.json

    x = data.get('x')
    y = data.get('y')
    z = data.get('z')

    if x is None or y is None or z is None:
        return jsonify({"error": "Missing coordinates. Please provide x, y, and z."}), 400

    results = []


    for blk in get_blocks():
        
        if blk['block_indices'][0] == x and blk['block_indices'][1] == y and blk['block_indices'][2] == z:
            results.append((blk))

    return jsonify({"results": results})




def run_flask_with_cloudflared():
    """
    Run the Flask application behind a Cloudflare tunnel for public sharing.
    The Flask app is served on the specified local port, and a public URL
    is generated via the Cloudflare tunnel.
    """
    cloudflared_port = 8000
    tunnel_url = _run_cloudflared(cloudflared_port, 7899)
    print(f"Cloudflare tunnel URL: {tunnel_url}")
    app.run(port=cloudflared_port)

def launch(share=False):
    """
    Launch the Flask application.
    
    If share=True:
        - Starts the server in a separate thread.
        - Establishes a Cloudflare tunnel for public access.
        - Attempts to retrieve and print the generated tunnel URL.
    Otherwise:
        - Runs the Flask app locally on port 5002.
    """
    print("Launching Triton viz tool")

    if share:
        # Run with Cloudflare tunnel on a separate thread.
        flask_thread = threading.Thread(target=run_flask_with_cloudflared)
        flask_thread.start()

        # Give the server some time to start up.
        time.sleep(2)

        # Try to get the tunnel URL by making a request to localhost.
        try:
            response = requests.get("http://localhost:8000")
            print(f"Your app is now available at: {response.url}")
        except requests.exceptions.RequestException:
            print("Please wait for URL:")
    else:
        # Run the Flask app locally without a tunnel.
        app.run(port=5002, host="0.0.0.0")

def stop_server():
    """
    Stop the Flask server if needed. Currently not implemented.
    """
    pass

if __name__ == "__main__":
    launch()
