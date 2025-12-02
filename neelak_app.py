"""
Neelakshi Kaundal - Ultimate Automation Dashboard
Comprehensive task automation with SSH support, Docker integration, and real-time terminal
"""

import streamlit as st
import subprocess
import os
import time
import requests
from bs4 import BeautifulSoup
import speech_recognition as sr
from openai import OpenAI
import git
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from io import BytesIO
import base64
from PIL import Image
import cv2
import paramiko
import socket
import select
import re
import json
import threading
import queue
import docker
import boto3
from botocore.exceptions import ClientError

# --- Configuration ---
st.set_page_config(
    page_title="Neelakshi Kaundal - Automation Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Global Variables ---
if 'terminal_output' not in st.session_state:
    st.session_state.terminal_output = []
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
if 'google_cse_id' not in st.session_state:
    st.session_state.google_cse_id = ""
if 'google_api_key' not in st.session_state:
    st.session_state.google_api_key = ""
if 'aws_access_key' not in st.session_state:
    st.session_state.aws_access_key = ""
if 'aws_secret_key' not in st.session_state:
    st.session_state.aws_secret_key = ""
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""
if 'ssh_host' not in st.session_state:
    st.session_state.ssh_host = "localhost"
if 'ssh_user' not in st.session_state:
    st.session_state.ssh_user = os.getenv("USER", "ubuntu")
if 'ssh_password' not in st.session_state:
    st.session_state.ssh_password = ""
if 'ssh_key_path' not in st.session_state:
    st.session_state.ssh_key_path = ""
if 'docker_client' not in st.session_state:
    st.session_state.docker_client = None

# --- Terminal Functions ---
def add_to_terminal(text, is_command=False):
    """Add text to terminal output"""
    prefix = "🚀 $ " if is_command else "     "
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.terminal_output.append(f"[{timestamp}] {prefix}{text}")
    # Keep only last 1000 lines to prevent memory issues
    if len(st.session_state.terminal_output) > 1000:
        st.session_state.terminal_output = st.session_state.terminal_output[-1000:]

def clear_terminal():
    """Clear terminal output"""
    st.session_state.terminal_output = []
    add_to_terminal("Terminal cleared. Ready for new commands.", is_command=False)

def display_terminal():
    """Display the terminal output"""
    st.markdown("### 🖥️ Live Terminal Output")
    terminal_text = "\n".join(st.session_state.terminal_output)
    st.text_area(
        "Terminal",
        terminal_text,
        height=400,
        key="terminal_display",
        label_visibility="collapsed"
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("🗑️ Clear Terminal", help="Click to clear all terminal output"):
            clear_terminal()
    with col2:
        if st.button("📋 Copy Output", help="Click to copy terminal output to clipboard"):
            st.code(terminal_text)
    with col3:
        if st.button("🔄 Refresh", help="Click to refresh terminal view"):
            st.rerun()

# --- Command Execution Functions ---
def run_command(command, task_name="", show_output=True, add_to_terminal_flag=True):
    """Execute command locally or via SSH"""
    try:
        if add_to_terminal_flag:
            add_to_terminal(f"Executing: {command}", is_command=True)

        if st.session_state.ssh_host and st.session_state.ssh_host != "localhost":
            return run_ssh_command(command, task_name, show_output, add_to_terminal_flag)
        else:
            return run_local_command(command, task_name, show_output, add_to_terminal_flag)
    except Exception as e:
        error_msg = f"Command execution failed: {str(e)}"
        if add_to_terminal_flag:
            add_to_terminal(f"❌ {error_msg}")
        st.error(error_msg)
        return ""

def run_local_command(command, task_name="", show_output=True, add_to_terminal_flag=True):
    """Execute command on local system"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)

        if add_to_terminal_flag:
            if result.stdout:
                add_to_terminal(result.stdout)
            if result.stderr:
                add_to_terminal(f"STDERR: {result.stderr}")

        if show_output:
            if result.returncode == 0:
                success_msg = f"✅ {task_name} completed successfully!" if task_name else "✅ Command executed successfully!"
                st.success(success_msg)
            else:
                error_msg = f"❌ {task_name} failed!" if task_name else "❌ Command failed!"
                st.error(error_msg)

        return result.stdout
    except subprocess.TimeoutExpired:
        error_msg = f"❌ {task_name} timed out!" if task_name else "❌ Command timed out!"
        if add_to_terminal_flag:
            add_to_terminal(error_msg)
        st.error(error_msg)
        return ""
    except Exception as e:
        error_msg = f"Local command error: {str(e)}"
        if add_to_terminal_flag:
            add_to_terminal(error_msg)
        st.error(error_msg)
        return ""

# --- SSH Helper Functions ---
def get_ssh_client():
    """Create and return an SSH client with proper configuration"""
    if st.session_state.ssh_host == "localhost":
        return None

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Try key-based authentication first
        if st.session_state.ssh_key_path and os.path.exists(st.session_state.ssh_key_path):
            key = paramiko.RSAKey.from_private_key_file(st.session_state.ssh_key_path)
            ssh.connect(
                hostname=st.session_state.ssh_host,
                username=st.session_state.ssh_user,
                pkey=key,
                timeout=10
            )
            return ssh

        # Fall back to password authentication
        if st.session_state.ssh_password:
            ssh.connect(
                hostname=st.session_state.ssh_host,
                username=st.session_state.ssh_user,
                password=st.session_state.ssh_password,
                timeout=10
            )
            return ssh

        # Try without password for local SSH
        ssh.connect(
            hostname=st.session_state.ssh_host,
            username=st.session_state.ssh_user,
            timeout=10
        )
        return ssh

    except Exception as e:
        error_msg = f"SSH connection failed: {str(e)}"
        add_to_terminal(f"❌ {error_msg}")
        st.error(error_msg)
        return None

def run_ssh_command(command, task_name="", show_output=True, add_to_terminal_flag=True):
    """Execute a command on the remote Linux machine via SSH"""
    ssh = get_ssh_client()
    if not ssh:
        return ""

    try:
        stdin, stdout, stderr = ssh.exec_command(command, timeout=120)
        output = stdout.read().decode()
        error = stderr.read().decode()

        if add_to_terminal_flag:
            if output:
                add_to_terminal(output)
            if error:
                add_to_terminal(f"STDERR: {error}")

        if show_output:
            if error:
                error_msg = f"❌ {task_name} failed: {error}" if task_name else f"❌ Command failed: {error}"
                st.error(error_msg)
            else:
                success_msg = f"✅ {task_name} completed successfully on remote host!" if task_name else "✅ Command executed successfully on remote host!"
                st.success(success_msg)

        return output
    except Exception as e:
        error_msg = f"SSH Error: {str(e)}"
        if add_to_terminal_flag:
            add_to_terminal(f"❌ {error_msg}")
        st.error(error_msg)
        return ""
    finally:
        ssh.close()

# --- Docker Management ---
def init_docker_client():
    """Initialize Docker client"""
    try:
        st.session_state.docker_client = docker.from_env()
        add_to_terminal("✅ Docker client initialized successfully")
        return True
    except Exception as e:
        add_to_terminal(f"❌ Docker client initialization failed: {str(e)}")
        st.session_state.docker_client = None
        return False

def pull_docker_image(image_name):
    """Pull Docker image with progress"""
    try:
        if not st.session_state.docker_client:
            if not init_docker_client():
                return False

        add_to_terminal(f"📥 Pulling Docker image: {image_name}")

        # Create a placeholder for progress
        progress_placeholder = st.empty()
        progress_placeholder.info(f"Pulling {image_name}... This may take a few minutes.")

        # Pull the image
        image = st.session_state.docker_client.images.pull(image_name)

        progress_placeholder.success(f"✅ Successfully pulled {image_name}")
        add_to_terminal(f"✅ Successfully pulled Docker image: {image_name}")
        return True
    except Exception as e:
        add_to_terminal(f"❌ Failed to pull Docker image: {str(e)}")
        st.error(f"Failed to pull Docker image: {str(e)}")
        return False

def run_docker_container(image_name, container_name="", ports=None, volumes=None):
    """Run Docker container"""
    try:
        if not st.session_state.docker_client:
            if not init_docker_client():
                return None

        add_to_terminal(f"🐳 Running Docker container from: {image_name}")

        container = st.session_state.docker_client.containers.run(
            image_name,
            name=container_name if container_name else None,
            ports=ports if ports else {},
            volumes=volumes if volumes else {},
            detach=True
        )

        add_to_terminal(f"✅ Docker container started: {container.id[:12]}")
        return container
    except Exception as e:
        add_to_terminal(f"❌ Failed to run Docker container: {str(e)}")
        st.error(f"Failed to run Docker container: {str(e)}")
        return None

# ========================
# TASK IMPLEMENTATIONS
# ========================

# --- Git Tasks ---
def git_tasks():
    st.header("🔧 Git Automation")

    with st.expander("📂 Repository Operations", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Create & Initialize Repository")
            repo_name = st.text_input("Repository Name", "my-project", key="git_repo_name")
            file_name = st.text_input("File Name", "README.md", key="git_file_name")
            commit_message = st.text_input("Commit Message", "Initial commit", key="git_commit_msg")

            if st.button("🚀 Create and Commit", key="git_create"):
                commands = [
                    f"mkdir -p {repo_name}",
                    f"cd {repo_name} && git init",
                    f"cd {repo_name} && echo '# {repo_name}' > {file_name}",
                    f"cd {repo_name} && git add .",
                    f"cd {repo_name} && git config user.email 'test@example.com'",
                    f"cd {repo_name} && git config user.name 'Test User'",
                    f"cd {repo_name} && git commit -m '{commit_message}'"
                ]
                for cmd in commands:
                    run_command(cmd, f"Executing: {cmd}")
                st.success(f"Repository '{repo_name}' created and committed!")

        with col2:
            st.subheader("Git Status & Operations")
            repo_path = st.text_input("Repository Path", ".", key="git_repo_path")

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("📊 Git Status", key="git_status"):
                    run_command("git status", "Git Status")
                if st.button("📜 Git Log", key="git_log"):
                    run_command("git log --oneline -10", "Recent Commits")
            with col_btn2:
                if st.button("🔄 Git Pull", key="git_pull"):
                    run_command("git pull", "Git Pull")
                if st.button("📤 Git Push", key="git_push"):
                    run_command("git push", "Git Push")

            clone_url = st.text_input("Clone URL", "https://github.com/username/repository.git", key="git_clone_url")
            if st.button("📥 Clone Repository", key="git_clone"):
                run_command(f"git clone {clone_url}", "Cloning Repository")

# --- Machine Learning Tasks ---
def ml_tasks():
    st.header("🤖 Machine Learning Automation")

    with st.expander("📊 Data Processing", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Install ML Packages")
            if st.button("📦 Install ML Dependencies", key="ml_install"):
                run_command("pip install scikit-learn pandas numpy matplotlib seaborn tensorflow torch", "Installing ML Packages")

            st.subheader("Create Sample Dataset")
            if st.button("📝 Generate Sample Data", key="ml_data"):
                # Create sample CSV data
                sample_data = """age,salary,department,experience
25,50000,IT,2
30,60000,HR,4
28,55000,Finance,3
35,70000,IT,8
40,80000,HR,10
32,65000,Finance,6
29,52000,IT,3
45,85000,HR,15"""

                with open("sample_data.csv", "w") as f:
                    f.write(sample_data)

                run_command("head -n 8 sample_data.csv", "Sample Data Created")

        with col2:
            st.subheader("Data Analysis")
            if st.button("📈 Analyze Dataset", key="ml_analyze"):
                commands = [
                    "echo '=== Dataset Info ==='",
                    "wc -l sample_data.csv",
                    "echo '=== First 5 rows ==='",
                    "head -n 5 sample_data.csv",
                    "echo '=== Basic Statistics ==='",
                    "python3 -c \"import pandas as pd; df = pd.read_csv('sample_data.csv'); print(df.describe())\""
                ]
                for cmd in commands:
                    run_command(cmd, f"Running: {cmd}")

    with st.expander("🧠 Model Training", expanded=True):
        st.subheader("Train Machine Learning Models")

        model_type = st.selectbox("Select Model Type",
                                ["Linear Regression", "Random Forest", "Neural Network"],
                                key="ml_model_type")

        if st.button("🎯 Train Model", key="ml_train"):
            if model_type == "Linear Regression":
                ml_script = """
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Create sample data
np.random.seed(42)
X = np.random.randn(100, 3)
y = X[:, 0] * 2 + X[:, 1] * 1.5 + X[:, 2] * 0.5 + np.random.randn(100) * 0.1

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
score = model.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)

print(f"Model R² score: {score:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_:.4f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Regression: Actual vs Predicted')
plt.savefig('linear_regression_results.png')
plt.close()

print("Plot saved as 'linear_regression_results.png'")
"""
            elif model_type == "Random Forest":
                ml_script = """
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Create sample data
np.random.seed(42)
X = np.random.randn(100, 3)
y = X[:, 0] * 2 + X[:, 1] * 1.5 + X[:, 2] * 0.5 + np.random.randn(100) * 0.1

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
score = model.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)

print(f"Random Forest R² score: {score:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Feature importances: {model.feature_importances_}")
"""
            else:  # Neural Network
                ml_script = """
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Create sample data
np.random.seed(42)
X = np.random.randn(100, 3)
y = X[:, 0] * 2 + X[:, 1] * 1.5 + X[:, 2] * 0.5 + np.random.randn(100) * 0.1

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                    random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
score = model.score(X_test_scaled, y_test)
mse = mean_squared_error(y_test, y_pred)

print(f"Neural Network R² score: {score:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Training iterations: {model.n_iter_}")
"""

            with open("ml_training.py", "w") as f:
                f.write(ml_script)

            run_command("python3 ml_training.py", f"Training {model_type} Model")

            # Show the generated plot if it exists
            if model_type == "Linear Regression" and os.path.exists("linear_regression_results.png"):
                st.image("linear_regression_results.png", caption="Linear Regression Results")

# --- Web Development Tasks ---
def webdev_tasks():
    st.header("🌐 Web Development Automation")

    with st.expander("🚀 Project Setup", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Create Web Project")
            project_name = st.text_input("Project Name", "my-webapp", key="web_project_name")
            project_type = st.selectbox("Project Type", ["HTML/CSS/JS", "Flask", "Django", "React"], key="web_project_type")

            if st.button("📁 Create Project Structure", key="web_create"):
                if project_type == "HTML/CSS/JS":
                    commands = [
                        f"mkdir -p {project_name}",
                        f"cd {project_name} && mkdir css js images",
                        f"cd {project_name} && echo '<!DOCTYPE html><html><head><title>{project_name}</title><link rel=\"stylesheet\" href=\"css/style.css\"></head><body><h1>Welcome to {project_name}</h1><script src=\"js/script.js\"></script></body></html>' > index.html",
                        f"cd {project_name} && echo 'body {{ font-family: Arial, sans-serif; margin: 40px; background: #f0f0f0; }} h1 {{ color: #333; }}' > css/style.css",
                        f"cd {project_name} && echo 'console.log(\"{project_name} loaded!\"); document.addEventListener(\"DOMContentLoaded\", function() {{ document.body.style.backgroundColor = \"#e0f7fa\"; }});' > js/script.js"
                    ]
                elif project_type == "Flask":
                    commands = [
                        f"mkdir -p {project_name}",
                        f"cd {project_name} && mkdir templates static",
                        f"cd {project_name} && echo 'from flask import Flask, render_template\\napp = Flask(__name__)\\n\\n@app.route(\"/\")\\ndef home():\\n    return render_template(\\\"index.html\\\", title=\\\"{project_name}\\\")\\n\\nif __name__ == \\\"__main__\\\":\\n    app.run(debug=True, host=\\\"0.0.0.0\\\")' > app.py",
                        f"cd {project_name}/templates && echo '<!DOCTYPE html><html><head><title>{{ title }}</title></head><body><h1>Welcome to {{ title }}</h1><p>This is a Flask application.</p></body></html>' > index.html"
                    ]
                elif project_type == "React":
                    commands = [
                        f"npx create-react-app {project_name}",
                        f"cd {project_name} && npm install"
                    ]

                for cmd in commands:
                    run_command(cmd, f"Setting up: {cmd}")

                st.success(f"Project '{project_name}' created!")

        with col2:
            st.subheader("Web Server")
            port = st.number_input("Port", 8080, min_value=1000, max_value=9999, key="web_port")

            col_web1, col_web2 = st.columns(2)
            with col_web1:
                if st.button("🌐 Start HTTP Server", key="web_start"):
                    if project_type == "Flask":
                        run_command(f"cd {project_name} && python3 app.py &", "Starting Flask Server")
                    else:
                        run_command(f"cd {project_name} && python3 -m http.server {port} &", "Starting Web Server")
                    st.success(f"Server running at http://localhost:{port}")

                if st.button("🛑 Stop Servers", key="web_stop"):
                    run_command("pkill -f 'python.*http.server' || pkill -f 'python.*app.py' || echo 'No servers running'", "Stopping Servers")

            with col_web2:
                if st.button("📊 Check Running Servers", key="web_check"):
                    run_command("netstat -tulpn | grep python || ps aux | grep python | head -10", "Checking Servers")

    with st.expander("🔗 API Testing", expanded=True):
        st.subheader("Test REST API")
        api_url = st.text_input("API URL", "https://jsonplaceholder.typicode.com/posts/1", key="api_url")

        if st.button("🧪 Test API", key="api_test"):
            try:
                add_to_terminal(f"Testing API: {api_url}")
                response = requests.get(api_url, timeout=10)
                if response.status_code == 200:
                    st.success("✅ API is working!")
                    st.json(response.json())
                    add_to_terminal(f"API Response: Status {response.status_code}")
                else:
                    st.error(f"❌ API returned status: {response.status_code}")
                    add_to_terminal(f"API Error: Status {response.status_code}")
            except Exception as e:
                st.error(f"API test failed: {str(e)}")
                add_to_terminal(f"API Test Failed: {str(e)}")

# --- Docker Management ---
def docker_tasks():
    st.header("🐳 Docker Management")

    with st.expander("🚀 Docker Operations", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Image Management")
            image_name = st.text_input("Image Name", "nginx:latest", key="docker_image")
            container_name = st.text_input("Container Name", "my-container", key="docker_container")

            col_img1, col_img2 = st.columns(2)
            with col_img1:
                if st.button("📥 Pull Image", key="docker_pull"):
                    if pull_docker_image(image_name):
                        st.success(f"✅ Image {image_name} pulled successfully!")

                if st.button("🐳 Run Container", key="docker_run"):
                    container = run_docker_container(
                        image_name,
                        container_name=container_name,
                        ports={'80/tcp': 8080}
                    )
                    if container:
                        st.success(f"✅ Container {container.id[:12]} started!")

            with col_img2:
                if st.button("📋 List Images", key="docker_images"):
                    run_command("docker images", "Docker Images")

                if st.button("🛑 Stop Container", key="docker_stop"):
                    run_command(f"docker stop {container_name}", f"Stopping {container_name}")

        with col2:
            st.subheader("Container Management")
            col_cont1, col_cont2 = st.columns(2)

            with col_cont1:
                if st.button("📊 Running Containers", key="docker_ps"):
                    run_command("docker ps", "Running Containers")

                if st.button("🗑️ Remove Container", key="docker_rm"):
                    run_command(f"docker rm {container_name}", f"Removing {container_name}")

            with col_cont2:
                if st.button("📋 All Containers", key="docker_ps_a"):
                    run_command("docker ps -a", "All Containers")

                if st.button("🧹 System Prune", key="docker_prune"):
                    run_command("docker system prune -f", "Cleaning Docker System")

    with st.expander("📝 Docker Compose", expanded=True):
        st.subheader("Docker Compose Operations")
        compose_file = st.text_input("Compose File", "docker-compose.yml", key="compose_file")

        if st.button("📄 Create Sample Compose", key="docker_compose_create"):
            compose_content = """
version: '3'
services:
  web:
    image: nginx:latest
    ports:
      - "80:80"
    volumes:
      - ./html:/usr/share/nginx/html
  db:
    image: postgres:13
    environment:
      POSTGRES_PASSWORD: example
      POSTGRES_USER: user
      POSTGRES_DB: mydb
    ports:
      - "5432:5432"
"""
            with open(compose_file, "w") as f:
                f.write(compose_content)
            run_command(f"cat {compose_file}", f"Created {compose_file}")
            st.success(f"Sample {compose_file} created!")

        col_comp1, col_comp2, col_comp3 = st.columns(3)
        with col_comp1:
            if st.button("🚀 Compose Up", key="compose_up"):
                run_command(f"docker-compose -f {compose_file} up -d", "Starting Services")
        with col_comp2:
            if st.button("🛑 Compose Down", key="compose_down"):
                run_command(f"docker-compose -f {compose_file} down", "Stopping Services")
        with col_comp3:
            if st.button("📜 Compose Logs", key="compose_logs"):
                run_command(f"docker-compose -f {compose_file} logs", "Service Logs")

# --- System Management Tasks ---
def system_tasks():
    st.header("💻 System Management")

    with st.expander("📊 System Information", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🖥️ System Info", key="sys_info"):
                commands = [
                    "echo '=== System Information ==='",
                    "uname -a",
                    "echo '=== Disk Usage ==='",
                    "df -h | head -10",
                    "echo '=== Memory Usage ==='",
                    "free -h"
                ]
                for cmd in commands:
                    run_command(cmd, f"Running: {cmd}")

        with col2:
            if st.button("⚡ Process Info", key="sys_process"):
                run_command("ps aux --sort=-%cpu | head -10", "Top Processes")
                run_command("top -bn1 | head -20", "System Load")

        with col3:
            if st.button("🌐 Network Info", key="sys_network"):
                run_command("ifconfig || ip addr", "Network Configuration")
                run_command("netstat -tulpn | head -20", "Network Connections")

    with st.expander("📦 Package Management", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            package_name = st.text_input("Package Name", "curl", key="pkg_name")
            col_pkg1, col_pkg2 = st.columns(2)
            with col_pkg1:
                if st.button("📥 Install Package", key="pkg_install"):
                    run_command(f"sudo apt-get install -y {package_name}", f"Installing {package_name}")
            with col_pkg2:
                if st.button("🔍 Search Package", key="pkg_search"):
                    run_command(f"apt-cache search {package_name}", f"Searching {package_name}")

        with col2:
            col_pkg3, col_pkg4 = st.columns(2)
            with col_pkg3:
                if st.button("🔄 Update System", key="pkg_update"):
                    run_command("sudo apt-get update", "Updating Package List")
            with col_pkg4:
                if st.button("⚡ Upgrade System", key="pkg_upgrade"):
                    run_command("sudo apt-get upgrade -y", "Upgrading Packages")

    with st.expander("📁 File Operations", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            filename = st.text_input("File to Create", "test.txt", key="file_name")
            content = st.text_area("File Content", "Hello, World! This is a test file.", key="file_content")

            col_file1, col_file2 = st.columns(2)
            with col_file1:
                if st.button("📝 Create File", key="file_create"):
                    run_command(f"echo '{content}' > {filename}", f"Creating {filename}")

                if st.button("📋 List Files", key="file_list"):
                    run_command("ls -la", "Directory Listing")
            with col_file2:
                if st.button("👀 View File", key="file_view"):
                    run_command(f"cat {filename}", f"Viewing {filename}")

                if st.button("🗑️ Delete File", key="file_delete"):
                    run_command(f"rm {filename}", f"Deleting {filename}")

        with col2:
            search_term = st.text_input("Search Term", "test", key="search_term")
            search_path = st.text_input("Search Path", ".", key="search_path")

            if st.button("🔍 Search Files", key="file_search"):
                run_command(f"find {search_path} -name '*{search_term}*' -type f", f"Searching for {search_term}")

# --- Database Operations ---
def database_tasks():
    st.header("🗄️ Database Operations")

    with st.expander("🐬 MySQL Operations", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("MySQL Management")
            col_mysql1, col_mysql2 = st.columns(2)
            with col_mysql1:
                if st.button("📊 MySQL Status", key="mysql_status"):
                    run_command("systemctl status mysql || service mysql status", "MySQL Status")
                if st.button("🚀 Start MySQL", key="mysql_start"):
                    run_command("sudo systemctl start mysql || sudo service mysql start", "Starting MySQL")
            with col_mysql2:
                if st.button("🛑 Stop MySQL", key="mysql_stop"):
                    run_command("sudo systemctl stop mysql || sudo service mysql stop", "Stopping MySQL")
                if st.button("🔄 Restart MySQL", key="mysql_restart"):
                    run_command("sudo systemctl restart mysql || sudo service mysql restart", "Restarting MySQL")

            db_name = st.text_input("Database Name", "testdb", key="mysql_db")
            if st.button("📁 Create Database", key="mysql_create_db"):
                run_command(f"sudo mysql -e 'CREATE DATABASE IF NOT EXISTS {db_name};'", f"Creating database {db_name}")

        with col2:
            st.subheader("MySQL Queries")
            query = st.text_input("MySQL Query", "SHOW DATABASES;", key="mysql_query")
            col_query1, col_query2 = st.columns(2)
            with col_query1:
                if st.button("⚡ Execute Query", key="mysql_execute"):
                    run_command(f"sudo mysql -e \"{query}\"", "Executing MySQL Query")
            with col_query2:
                if st.button("📋 Show Tables", key="mysql_tables"):
                    run_command(f"sudo mysql -e 'SHOW TABLES FROM {db_name};'", "Showing Tables")

    with st.expander("🍃 MongoDB Operations", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📊 MongoDB Status", key="mongo_status"):
                run_command("systemctl status mongod || service mongod status", "MongoDB Status")
            if st.button("🚀 Start MongoDB", key="mongo_start"):
                run_command("sudo systemctl start mongod || sudo service mongod start", "Starting MongoDB")

        with col2:
            if st.button("🛑 Stop MongoDB", key="mongo_stop"):
                run_command("sudo systemctl stop mongod || sudo service mongod stop", "Stopping MongoDB")
            if st.button("🔄 Restart MongoDB", key="mongo_restart"):
                run_command("sudo systemctl restart mongod || sudo service mongod restart", "Restarting MongoDB")

# --- AI & Chat Integration ---
def ai_tasks():
    st.header("🧠 AI & Chat Integration")

    with st.expander("🔑 API Configuration", expanded=True):
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.openai_api_key,
            help="Get your API key from https://platform.openai.com/api-keys"
        )

        st.session_state.gemini_api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            value=st.session_state.gemini_api_key,
            help="Get your API key from https://aistudio.google.com/app/apikey"
        )

    with st.expander("💬 ChatGPT Integration", expanded=True):
        if not st.session_state.openai_api_key:
            st.warning("⚠️ Please enter your OpenAI API key above to use ChatGPT features")
        else:
            client = OpenAI(api_key=st.session_state.openai_api_key)

            prompt = st.text_area(
                "ChatGPT Prompt",
                "Explain quantum computing in simple terms",
                height=100
            )

            col_ai1, col_ai2 = st.columns(2)
            with col_ai1:
                if st.button("🤖 Generate Response", key="chatgpt_generate"):
                    with st.spinner("Generating response..."):
                        try:
                            response = client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[{"role": "user", "content": prompt}],
                                max_tokens=500
                            )
                            st.success("✅ Response generated!")
                            st.write(response.choices[0].message.content)
                            add_to_terminal("ChatGPT response generated successfully")
                        except Exception as e:
                            st.error(f"❌ API Error: {str(e)}")
                            add_to_terminal(f"ChatGPT Error: {str(e)}")

            with col_ai2:
                if st.button("🎨 Generate Image", key="dalle_generate"):
                    image_prompt = st.text_input("Image Prompt", "A futuristic cityscape at sunset", key="image_prompt")
                    with st.spinner("Creating image..."):
                        try:
                            response = client.images.generate(
                                model="dall-e-2",
                                prompt=image_prompt,
                                size="512x512",
                                quality="standard",
                                n=1,
                            )
                            image_url = response.data[0].url
                            st.image(image_url, caption=image_prompt)
                            add_to_terminal("DALL-E image generated successfully")
                        except Exception as e:
                            st.error(f"❌ Image generation failed: {str(e)}")
                            add_to_terminal(f"DALL-E Error: {str(e)}")

    with st.expander("🎤 Speech Recognition", expanded=True):
        st.info("Click below to record and transcribe speech")

        if st.button("🎤 Start Recording", key="speech_record"):
            try:
                recognizer = sr.Recognizer()
                with sr.Microphone() as source:
                    st.info("Listening... Speak now!")
                    audio = recognizer.listen(source, timeout=10)

                    text = recognizer.recognize_google(audio)
                    st.success("✅ Transcription complete!")
                    st.text_area("Transcribed Text", text, height=100)
                    add_to_terminal(f"Speech transcribed: {text}")
            except sr.UnknownValueError:
                st.error("❌ Could not understand audio")
                add_to_terminal("Speech recognition failed: Could not understand audio")
            except sr.RequestError as e:
                st.error(f"❌ Recognition error: {e}")
                add_to_terminal(f"Speech recognition error: {e}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                add_to_terminal(f"Speech recognition error: {str(e)}")

# --- Main App Structure ---
def main():
    # Sidebar
    st.sidebar.title("Neelakshi Kaundal")
    st.sidebar.markdown("### 🚀 Ultimate Automation Dashboard")
    st.sidebar.markdown("---")

    # SSH Configuration
    with st.sidebar.expander("🔑 SSH Configuration", expanded=True):
        st.session_state.ssh_host = st.text_input(
            "SSH Host",
            value=st.session_state.ssh_host,
            help="Enter 'localhost' for local execution or remote host IP"
        )
        st.session_state.ssh_user = st.text_input(
            "SSH Username",
            value=st.session_state.ssh_user
        )
        st.session_state.ssh_password = st.text_input(
            "SSH Password",
            type="password",
            value=st.session_state.ssh_password
        )
        st.session_state.ssh_key_path = st.text_input(
            "SSH Key Path (optional)",
            value=st.session_state.ssh_key_path,
            help="Path to SSH private key file"
        )

        col_ssh1, col_ssh2 = st.columns(2)
        with col_ssh1:
            if st.button("🔍 Test SSH", key="ssh_test"):
                if st.session_state.ssh_host and st.session_state.ssh_host != "localhost":
                    ssh = get_ssh_client()
                    if ssh:
                        st.success("✅ SSH connection successful!")
                        ssh.close()
                        add_to_terminal("SSH connection test: SUCCESS")
                    else:
                        st.error("❌ SSH connection failed")
                        add_to_terminal("SSH connection test: FAILED")
                else:
                    st.success("✅ Using local system")
                    add_to_terminal("Using local system execution")
        with col_ssh2:
            if st.button("🔄 Reset SSH", key="ssh_reset"):
                st.session_state.ssh_host = "localhost"
                st.session_state.ssh_user = os.getenv("USER", "ubuntu")
                st.session_state.ssh_password = ""
                st.session_state.ssh_key_path = ""
                st.rerun()

    # Navigation
    st.sidebar.markdown("---")
    page = st.sidebar.selectbox("🎯 Select Category", [
        "🏠 Dashboard",
        "🔧 Git Tasks",
        "🤖 Machine Learning",
        "🌐 Web Development",
        "🐳 Docker Management",
        "💻 System Management",
        "🗄️ Database Operations",
        "🧠 AI Integration"
    ])

    st.sidebar.markdown("---")
    st.sidebar.info("💡 All tasks execute on your system with real-time terminal output")

    # Main content area with two columns
    col_main, col_terminal = st.columns([7, 3])

    with col_main:
        # Page Header
        if page != "🏠 Dashboard":
            st.title(f"{page.split(' ')[1]} Automation")
            current_host = st.session_state.ssh_host if st.session_state.ssh_host else "localhost"
            st.subheader(f"🌍 Executing on: {current_host}")
        else:
            st.title("🚀 Ultimate Automation Dashboard")
            st.subheader("Welcome to Neelakshi Kaundal's Comprehensive Automation Platform")

        st.markdown("---")

        # Page Routing
        if page == "🏠 Dashboard":
            show_dashboard()
        elif page == "🔧 Git Tasks":
            git_tasks()
        elif page == "🤖 Machine Learning":
            ml_tasks()
        elif page == "🌐 Web Development":
            webdev_tasks()
        elif page == "🐳 Docker Management":
            docker_tasks()
        elif page == "💻 System Management":
            system_tasks()
        elif page == "🗄️ Database Operations":
            database_tasks()
        elif page == "🧠 AI Integration":
            ai_tasks()

    with col_terminal:
        display_terminal()

def show_dashboard():
    """Show the main dashboard with overview and quick actions"""

    st.markdown("""
    ## 📊 Dashboard Overview

    This comprehensive automation dashboard provides powerful tools for:
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🔧 Development
        - Git repository management
        - Web project scaffolding
        - Docker container management
        - Database operations
        """)

    with col2:
        st.markdown("""
        ### 🤖 AI & ML
        - Machine learning workflows
        - Data processing pipelines
        - Model training automation
        - AI integration (ChatGPT, Gemini)
        """)

    with col3:
        st.markdown("""
        ### 💻 System Admin
        - System monitoring
        - Package management
        - File operations
        - Security scanning
        """)

    st.markdown("---")

    # Quick Actions
    st.subheader("⚡ Quick Actions")

    col_q1, col_q2, col_q3, col_q4 = st.columns(4)

    with col_q1:
        if st.button("🔄 System Update", key="quick_update"):
            run_command("sudo apt-get update && sudo apt-get upgrade -y", "System Update")

    with col_q2:
        if st.button("📊 System Info", key="quick_info"):
            run_command("uname -a && df -h && free -h", "System Information")

    with col_q3:
        if st.button("🐳 Docker Status", key="quick_docker"):
            run_command("docker --version && docker ps", "Docker Status")

    with col_q4:
        if st.button("🔍 Process Monitor", key="quick_process"):
            run_command("ps aux --sort=-%cpu | head -10", "Process Monitor")

    st.markdown("---")

    # Getting Started Guide
    with st.expander("📚 Getting Started Guide", expanded=True):
        st.markdown("""
        ### 🚀 First Time Setup

        1. **SSH Configuration** (Optional):
           - For remote execution, configure SSH credentials in the sidebar
           - Test connection with the "Test SSH" button

        2. **API Keys** (For AI Features):
           - OpenAI API key for ChatGPT and DALL-E
           - Google Gemini API key for AI features
           - Configure in the AI Integration section

        3. **Docker Setup**:
           - Ensure Docker is installed and running
           - Use the Docker Management section to pull and run images

        4. **Terminal Output**:
           - All command outputs appear in the right-side terminal
           - Use Clear Terminal button to reset the output
           - Copy button to copy terminal content

        ### 💡 Pro Tips

        - Use the terminal to monitor all command executions in real-time
        - Most operations work both locally and via SSH
        - Docker operations automatically handle image pulling
        - Machine learning examples include complete data generation
        """)

# Initialize Docker client on startup
if st.session_state.docker_client is None:
    init_docker_client()

# Add welcome message to terminal
if len(st.session_state.terminal_output) == 0:
    add_to_terminal("🚀 Welcome to Ultimate Automation Dashboard!")
    add_to_terminal("💡 Ready to execute commands. Select a category from the sidebar to get started.")
    add_to_terminal("📖 Check the Dashboard for getting started guide.")

if __name__ == "__main__":
    main()
