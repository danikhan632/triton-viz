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

# Internal module imports
from .analysis import analyze_records
from .draw import get_visualization_data
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

def precompute_c_values(op_data):
    """
    Precompute dot products (C values) for the given operation data to speed up future lookups.
    """
    input_data = op_data['input_data']
    other_data = op_data['other_data']
    rows, inner_dim = input_data.shape
    cols = other_data.shape[1]

    precomputed = {}
    for i in range(rows):
        for j in range(cols):
            # Initialize an array where precomputed[(i,j)][k] will store the dot product 
            # of the first k elements of input_data[i] and other_data for the given indices.
            precomputed[(i, j)] = [0] * (inner_dim + 1)
            for k in range(1, inner_dim + 1):
                val = torch.dot(input_data[i, :k], other_data[:k, j]).item()
                precomputed[(i, j)][k] = val

    return precomputed

def update_global_data():
    """
    Update the global_data dictionary by performing data analysis, fetching visualization data,
    and computing tooltips. Also precompute values for dot operations.
    """
    global global_data, raw_tensor_data, precomputed_c_values
    analysis_data = analyze_records()
    viz_data = get_visualization_data()

    global_data = {
        "ops": {
            "visualization_data": viz_data["visualization_data"],
            "failures": viz_data["failures"],
            "kernel_src": viz_data["kernel_src"]
        }
    }

    raw_tensor_data = viz_data["raw_tensor_data"]

    # Precompute C values for each Dot operation
    precomputed_c_values = {}
    for uuid, op_data in raw_tensor_data.items():
        if 'input_data' in op_data and 'other_data' in op_data:
            precomputed_c_values[uuid] = precompute_c_values(op_data)

    # Convert analysis data to DataFrame for tooltips
    df = pd.DataFrame(analysis_data, columns=["Metric", "Value"])
    analysis_with_tooltip = get_tooltip_data(df)
    global_data["analysis"] = analysis_with_tooltip

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
def get_data():
    """
    Return the global data as JSON. If data is not computed yet, update it first.
    """
    global global_data
    if global_data is None:
        update_global_data()
    return jsonify(global_data)

@app.route('/api/update_data')
def update_data():
    """
    Force an update of the global data and return a success status.
    """
    update_global_data()
    return jsonify({"status": "Data updated successfully"})

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





def tensor_to_json(tensor):

    try:
        
        def serialize_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, tl.constexpr):
                return obj.value
            raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

        tensor_dict = {}
        
        # List of potential attributes
        attributes = ['dtype', 'shape', 'stride', 'element_size', 'data']
        
        for attr in attributes:
            try:
                value = getattr(tensor, attr)
                if attr == 'data' and isinstance(value, torch.Tensor):
                    tensor_dict[attr] = value.cpu().numpy().tolist()
                elif attr == 'dtype':
                    tensor_dict[attr] = str(value)
                else:
                    tensor_dict[attr] = value
            except AttributeError:
                # If the attribute doesn't exist, we simply skip it
                pass

        return json.dumps(tensor_dict, default=serialize_numpy)
    except Exception as e:
        print(e)


def numpy_to_python(obj):

    for key in obj:
        if key == 'changed_vars':
            for var_key, value in obj[key].items():
                if isinstance(value, np.ndarray):
                    obj[key][var_key] = value.tolist()
                elif isinstance(value, np.integer):
                    obj[key][var_key] = int(value)
                elif isinstance(value, np.floating):
                    obj[key][var_key] = float(value)
                elif isinstance(value, dict):
                    obj[key][var_key] = numpy_to_python(value)
                elif isinstance(value, list):
                    obj[key][var_key] = [numpy_to_python(item) for item in value]
                elif isinstance(value, triton.language.core.tensor):
 
                    obj[key][var_key] = value.to_json()
         

    return obj


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
