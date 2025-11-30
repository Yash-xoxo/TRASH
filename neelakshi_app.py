"""
Neelakshi Kaundal - Ultimate Automation Dashboard
Comprehensive task automation with SSH support for remote Linux execution
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
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, VideoTransformerBase
import av
import boto3
from botocore.exceptions import ClientError
import docker
from kubernetes import client, config
import yaml
import google.generativeai as genai
import paramiko
import socket
import select
import re

# --- Configuration ---
st.set_page_config(
    page_title="Neelakshi Kaundal - Automation Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API Keys Setup ---
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
    st.session_state.ssh_host = ""
if 'ssh_user' not in st.session_state:
    st.session_state.ssh_user = "root"
if 'ssh_password' not in st.session_state:
    st.session_state.ssh_password = ""
if 'ssh_key_path' not in st.session_state:
    st.session_state.ssh_key_path = ""

# --- SSH Helper Functions ---
def get_ssh_client():
    """Create and return an SSH client with proper configuration"""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    # Try key-based authentication first
    if st.session_state.ssh_key_path and os.path.exists(st.session_state.ssh_key_path):
        try:
            key = paramiko.RSAKey.from_private_key_file(st.session_state.ssh_key_path)
            ssh.connect(
                hostname=st.session_state.ssh_host,
                username=st.session_state.ssh_user,
                pkey=key,
                timeout=10
            )
            return ssh
        except Exception as e:
            st.warning(f"Key-based auth failed: {str(e)}")
    
    # Fall back to password authentication
    if st.session_state.ssh_password:
        try:
            ssh.connect(
                hostname=st.session_state.ssh_host,
                username=st.session_state.ssh_user,
                password=st.session_state.ssh_password,
                timeout=10
            )
            return ssh
        except Exception as e:
            st.error(f"SSH connection failed: {str(e)}")
            return None
    
    st.error("No SSH credentials provided")
    return None

def run_ssh_command(command, task_name=""):
    """Execute a command on the remote Linux machine via SSH"""
    ssh = get_ssh_client()
    if not ssh:
        return ""
    
    try:
        stdin, stdout, stderr = ssh.exec_command(command)
        output = stdout.read().decode()
        error = stderr.read().decode()
        
        if error:
            st.error(f"❌ {task_name} failed: {error}")
        else:
            st.success(f"✅ {task_name} completed successfully on remote host!")
        
        return output
    except Exception as e:
        st.error(f"SSH Error: {str(e)}")
        return ""
    finally:
        ssh.close()

def run_ssh_command_stream(command, task_name=""):
    """Execute a command with streaming output (for long-running tasks)"""
    ssh = get_ssh_client()
    if not ssh:
        return ""
    
    try:
        transport = ssh.get_transport()
        channel = transport.open_session()
        channel.exec_command(command)
        
        output = ""
        while True:
            if channel.exit_status_ready():
                break
            rl, wl, xl = select.select([channel], [], [], 0.0)
            if len(rl) > 0:
                recv = channel.recv(1024).decode()
                output += recv
                st.text(recv)
        
        exit_status = channel.recv_exit_status()
        if exit_status != 0:
            st.error(f"❌ {task_name} failed with exit code {exit_status}")
        else:
            st.success(f"✅ {task_name} completed successfully on remote host!")
        
        return output
    except Exception as e:
        st.error(f"SSH Error: {str(e)}")
        return ""
    finally:
        ssh.close()

# --- Docker Fix ---
def is_docker_installed():
    """Check if Docker is installed and available"""
    try:
        subprocess.run(["docker", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def is_docker_running():
    """Check if Docker daemon is running"""
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def get_docker_client():
    """Get Docker client with proper error handling"""
    if not is_docker_installed():
        st.error("Docker is not installed. Please install Docker.")
        return None
    
    if not is_docker_running():
        st.error("Docker daemon is not running. Please start Docker.")
        return None
    
    try:
        return docker.from_env()
    except docker.errors.DockerException as e:
        st.error(f"Docker connection failed: {str(e)}")
        return None

# ========================
# TASK IMPLEMENTATIONS
# ========================

# --- Git Tasks ---
def git_tasks():
    st.header("Git Automation")
    with st.expander("Create Git Repository", expanded=True):
        repo_name = st.text_input("Repository Name", "my-project")
        file_name = st.text_input("File Name", "README.md")
        commit_message = st.text_input("Commit Message", "Initial commit")
        
        if st.button("Create and Commit"):
            commands = [
                f"mkdir -p {repo_name}",
                f"cd {repo_name} && git init",
                f"cd {repo_name} && echo '# {repo_name}' > {file_name}",
                f"cd {repo_name} && git add .",
                f"cd {repo_name} && git commit -m '{commit_message}'"
            ]
            for cmd in commands:
                if st.session_state.ssh_host:
                    run_ssh_command(cmd)
                else:
                    run_command(cmd)
            st.success(f"Repository '{repo_name}' created and committed!")

# --- Machine Learning Tasks ---
def ml_tasks():
    st.header("Machine Learning Automation")
    
    # Task 1: Data Imputation Techniques
    with st.expander("Data Imputation Techniques", expanded=True):
        st.markdown("""
        **Common Data Imputation Techniques:**
        1. Mean/Median Imputation
        2. Mode Imputation (for categorical)
        3. K-Nearest Neighbors (KNN) Imputation
        4. Regression Imputation
        5. Multiple Imputation by Chained Equations (MICE)
        6. Deep Learning Methods (Autoencoders)
        """)
        if st.button("Show Example Code"):
            display_code("""
            from sklearn.impute import SimpleImputer, KNNImputer
            
            # Mean imputation
            mean_imputer = SimpleImputer(strategy='mean')
            data_imputed = mean_imputer.fit_transform(data)
            
            # KNN imputation
            knn_imputer = KNNImputer(n_neighbors=5)
            data_imputed = knn_imputer.fit_transform(data)
            """)
    
    # Task 2: Weight of Dropped Category
    with st.expander("Dropped Category Weight"):
        st.markdown("""
        **In categorical encoding:**
        - When using one-hot encoding, we drop one category to avoid multicollinearity
        - The weight of the dropped category is distributed among the remaining categories
        - The dropped category becomes the reference point (implicit weight=0)
        - Coefficients for other categories are interpreted relative to the dropped category
        """)
    
    # Task 3: Initializers and Use Cases
    with st.expander("Weight Initializers"):
        st.markdown("""
        **Common Initializers:**
        1. **Zeros**: Not recommended - causes symmetry problems
        2. **Random Normal**: Small random values from normal distribution
        3. **Xavier/Glorot**: Good for tanh and sigmoid (symmetric activations)
        4. **He**: Better for ReLU and its variants
        5. **LeCun**: For SELU activations
        
        **Use Cases:**
        - Use Xavier for tanh/sigmoid
        - Use He for ReLU/LeakyReLU
        - Use LeCun for SELU
        """)
    
    # Task 4: LLM Model Exploration
    with st.expander("LLM Model Exploration"):
        model_name = st.selectbox("Select LLM", ["GPT-2", "BERT", "LLaMA"])
        
        if st.button("Analyze Model Structure"):
            if model_name == "GPT-2":
                st.markdown("""
                **GPT-2 Architecture:**
                - Layers: Transformer decoder blocks
                - Neurons: 117M to 1.5B parameters
                - Activations: GeLU
                - Attention: Multi-head self-attention
                """)
                
                # Load and display model info
                tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                model = GPT2LMHeadModel.from_pretrained('gpt2')
                
                st.write("**Tokenizer:**", tokenizer)
                st.write("**Model:**", model)
                
                # Show sample generation
                input_text = "Artificial intelligence is"
                input_ids = tokenizer.encode(input_text, return_tensors='pt')
                output = model.generate(input_ids, max_length=50)
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                st.text_area("Generated Text", generated_text, height=100)
    
    # Task 5: Optimizers Use Cases
    with st.expander("Optimizers Guide"):
        st.markdown("""
        **Optimizer Use Cases:**
        1. **SGD**: Good for convex problems, requires tuning learning rate
        2. **Adam**: Default for most problems, adaptive learning rate
        3. **RMSprop**: Good for RNNs and non-stationary objectives
        4. **Adagrad**: Sparse data applications
        5. **Nadam**: Adam + Nesterov momentum
        """)
    
    # Task 6: Activation & Pooling Relationships
    with st.expander("Activation & Pooling"):
        st.markdown("""
        **Compatible Combinations:**
        | Activation      | Pooling        | Use Case                |
        |----------------|----------------|-------------------------|
        | ReLU           | Max Pooling    | CNNs (most common)      |
        | Sigmoid        | Average Pooling| Early CNN architectures|
        | Tanh           | Max Pooling    | RNNs, some CNNs         |
        | Leaky ReLU     | Max Pooling    | Prevents dead neurons  |
        | Softmax        | Global Average | Classification layers   |
        """)

# --- Web Development Tasks ---

def display_code(code_string):
    """Prints formatted code to the console for clarity."""
    print("\n" + "="*50)
    print("CODE SNIPPET:")
    print("="*50)
    print(code_string)
    print("="*50 + "\n")

def webdev_tasks():
    st.header("Web Development Automation")
    
    # Task 1: Speech to Text
    with st.expander("Speech to Text", expanded=True):
        st.info("Click start and speak into your microphone")
        recognizer = sr.Recognizer()
        text_output = st.empty()
        
        def recognize_speech():
            with sr.Microphone() as source:
                audio = recognizer.listen(source)
                try:
                    text = recognizer.recognize_google(audio)
                    text_output.text_area("Transcribed Text", text, height=100)
                except sr.UnknownValueError:
                    text_output.error("Could not understand audio")
                except sr.RequestError as e:
                    text_output.error(f"Recognition error: {e}")
        
        if st.button("Start Recording"):
            with st.spinner("Listening..."):
                recognize_speech()
    
    # Task 2: Camera Photo Capture
    with st.expander("Camera Photo Capture"):
        class VideoTransformer(VideoTransformerBase):
            def __init__(self):
                self.frame = None
            
            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")
                self.frame = img
        
        ctx = webrtc_streamer(
            key="camera",
            video_processor_factory=VideoTransformer,
            media_stream_constraints={"video": True, "audio": False}
        )
        
        if st.button("Capture Photo") and ctx.video_processor:
            img = Image.fromarray(ctx.video_processor.frame)
            st.image(img, caption="Captured Photo")
    
    # Task 3: Camera Live Stream
    with st.expander("Live Camera Stream"):
        class LiveStreamTransformer(VideoTransformerBase):
            def transform(self, frame):
                return frame
        
        webrtc_streamer(
            key="live-stream",
            video_processor_factory=LiveStreamTransformer,
            media_stream_constraints={"video": True, "audio": False}
        )
    
    # Task 4: Video Recording (Conceptual)
    with st.expander("Video Recording"):
        st.info("Full implementation requires Instagram API access")
        st.markdown("""
        **Conceptual Steps:**
        1. Record video using MediaRecorder API
        2. Save video to blob storage
        3. Use Instagram Graph API to post
        """)
        
        display_code("""
        // JavaScript Pseudocode
        const stream = await navigator.mediaDevices.getUserMedia({video: true});
        const recorder = new MediaRecorder(stream);
        
        recorder.ondataavailable = (e) => {
            const videoBlob = new Blob([e.data], {type: 'video/mp4'});
            
            // Upload to Instagram
            const formData = new FormData();
            formData.append('video', videoBlob);
            
            fetch('https://graph.instagram.com/me/media', {
                method: 'POST',
                headers: {'Authorization': 'Bearer ACCESS_TOKEN'},
                body: formData
            });
        };
        
        recorder.start();
        setTimeout(() => recorder.stop(), 10000); // Record 10s
        """, "javascript")
    
    # Task 5: Name Search Engine
    with st.expander("Name Search Engine"):
        name = st.text_input("Enter Name", "Neelakshi")
        
        if st.button("Search") and st.session_state.google_api_key and st.session_state.google_cse_id:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'q': name,
                'key': st.session_state.google_api_key,
                'cx': st.session_state.google_cse_id
            }
            
            try:
                response = requests.get(url, params=params).json()
                if 'items' in response:
                    st.subheader(f"Search Results for {name}")
                    for i, item in enumerate(response['items'][:5]):
                        st.markdown(f"{i+1}. [{item['title']}]({item['link']})")
                        st.caption(item['snippet'])
                else:
                    st.error("No results found")
            except Exception as e:
                st.error(f"Search failed: {str(e)}")
    
    # Task 7: Drag and Drop
    with st.expander("Drag and Drop"):
        st.markdown("""
        **HTML/CSS/JavaScript Implementation:**
        """)
        display_code("""
        <!DOCTYPE html>
        <html>
        <style>
            #div1, #div2 {
                width: 200px;
                height: 100px;
                padding: 10px;
                border: 1px solid #aaaaaa;
            }
        </style>
        <body>
            <div id="div1" ondrop="drop(event)" ondragover="allowDrop(event)">
                <p draggable="true" ondragstart="drag(event)" id="drag1">Drag me!</p>
            </div>
            <div id="div2" ondrop="drop(event)" ondragover="allowDrop(event)"></div>
            
            <script>
            function allowDrop(ev) {
                ev.preventDefault();
            }
            
            function drag(ev) {
                ev.dataTransfer.setData("text", ev.target.id);
            }
            
            function drop(ev) {
                ev.preventDefault();
                var data = ev.dataTransfer.getData("text");
                ev.target.appendChild(document.getElementById(data));
            }
            </script>
        </body>
        </html>
        """, "html")
    
    # Task 8-10: ChatGPT Integration
    with st.expander("ChatGPT Integration"):
        st.session_state.openai_api_key = st.text_input("OpenAI API Key", 
                                                      type="password",
                                                      value=st.session_state.openai_api_key)
        
        if not st.session_state.openai_api_key:
            st.warning("Enter OpenAI API key to continue")
            return
            
        client = OpenAI(api_key=st.session_state.openai_api_key)
        
        # Task 8: Basic ChatGPT
        prompt = st.text_area("ChatGPT Prompt", "Explain quantum computing in simple terms")
        
        if st.button("Generate Response"):
            with st.spinner("Generating response..."):
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    st.write(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"API Error: {str(e)}")
        
        # Task 9: Voice to ChatGPT
        st.subheader("Voice to ChatGPT")
        recognizer = sr.Recognizer()
        
        def voice_to_chatgpt():
            with sr.Microphone() as source:
                audio = recognizer.listen(source)
                try:
                    text = recognizer.recognize_google(audio)
                    st.session_state.voice_prompt = text
                except sr.UnknownValueError:
                    st.error("Could not understand audio")
                except sr.RequestError as e:
                    st.error(f"Recognition error: {e}")
        
        if st.button("Speak Prompt"):
            with st.spinner("Listening..."):
                voice_to_chatgpt()
        
        if 'voice_prompt' in st.session_state:
            st.text_area("Voice Prompt", st.session_state.voice_prompt)
            
            if st.button("Process Voice Prompt"):
                with st.spinner("Generating response..."):
                    try:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": st.session_state.voice_prompt}]
                        )
                        st.write(response.choices[0].message.content)
                    except Exception as e:
                        st.error(f"API Error: {str(e)}")
        
        # Task 10: Image Generation
        st.subheader("Image Generation")
        image_prompt = st.text_input("Image Generation Prompt", "A futuristic cityscape at sunset")
        
        if st.button("Generate Image"):
            with st.spinner("Creating image..."):
                try:
                    response = client.images.generate(
                        model="dall-e-3",
                        prompt=image_prompt,
                        size="1024x1024",
                        quality="standard",
                        n=1,
                    )
                    image_url = response.data[0].url
                    st.image(image_url, caption=image_prompt)
                except Exception as e:
                    st.error(f"Image generation failed: {str(e)}")



# --- AWS EC2 Management ---
def aws_ec2_tasks():
    st.header("AWS EC2 Management")
    
    # AWS Credentials
    with st.expander("AWS Credentials", expanded=True):
        st.session_state.aws_access_key = st.text_input("AWS Access Key", 
                                                      type="password",
                                                      value=st.session_state.aws_access_key)
        st.session_state.aws_secret_key = st.text_input("AWS Secret Key", 
                                                      type="password",
                                                      value=st.session_state.aws_secret_key)
        aws_region = st.selectbox("AWS Region", ["us-east-1", "us-west-2", "eu-west-1", "ap-south-1"])
    
    if not st.session_state.aws_access_key or not st.session_state.aws_secret_key:
        st.warning("Enter AWS credentials to continue")
        return
    
    ec2 = boto3.client(
        'ec2',
        aws_access_key_id=st.session_state.aws_access_key,
        aws_secret_access_key=st.session_state.aws_secret_key,
        region_name=aws_region
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("EC2 Instance Operations")
        instance_id = st.text_input("Instance ID")
        
        if st.button("Launch New Instance", key="launch_ec2"):
            try:
                response = ec2.run_instances(
                    ImageId='ami-0c55b159cbfafe1f0',  # Amazon Linux 2 AMI
                    InstanceType='t2.micro',
                    MinCount=1,
                    MaxCount=1
                )
                instance_id = response['Instances'][0]['InstanceId']
                st.success(f"Instance {instance_id} launched successfully!")
            except ClientError as e:
                st.error(f"AWS Error: {e.response['Error']['Message']}")
        
        if st.button("Start Instance", key="start_ec2") and instance_id:
            try:
                ec2.start_instances(InstanceIds=[instance_id])
                st.success(f"Instance {instance_id} starting...")
            except ClientError as e:
                st.error(f"AWS Error: {e.response['Error']['Message']}")
        
        if st.button("Stop Instance", key="stop_ec2") and instance_id:
            try:
                ec2.stop_instances(InstanceIds=[instance_id])
                st.success(f"Instance {instance_id} stopping...")
            except ClientError as e:
                st.error(f"AWS Error: {e.response['Error']['Message']}")
        
        if st.button("Terminate Instance", key="terminate_ec2") and instance_id:
            try:
                ec2.terminate_instances(InstanceIds=[instance_id])
                st.success(f"Instance {instance_id} terminating...")
            except ClientError as e:
                st.error(f"AWS Error: {e.response['Error']['Message']}")
    
    with col2:
        st.subheader("EC2 Instance Status")
        if st.button("Refresh Instance List"):
            try:
                response = ec2.describe_instances()
                instances = []
                for reservation in response['Reservations']:
                    for instance in reservation['Instances']:
                        instances.append({
                            "ID": instance['InstanceId'],
                            "Type": instance['InstanceType'],
                            "State": instance['State']['Name'],
                            "Public IP": instance.get('PublicIpAddress', 'N/A'),
                            "Launch Time": str(instance['LaunchTime'])
                        })
                if instances:
                    st.dataframe(pd.DataFrame(instances))
                else:
                    st.info("No EC2 instances found")
            except ClientError as e:
                st.error(f"AWS Error: {e.response['Error']['Message']}")

# --- Docker Management ---
def docker_tasks():
    st.header("Docker Management")
    
    if st.session_state.ssh_host:
        st.info(f"Running Docker commands on remote host: {st.session_state.ssh_host}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Image Management")
            image_name = st.text_input("Image Name", "nginx:latest")
            
            if st.button("Pull Image"):
                run_ssh_command(f"docker pull {image_name}", f"Pull {image_name}")
            
            if st.button("List Images"):
                output = run_ssh_command("docker images", "List Docker Images")
                st.text(output)
            
            if st.button("Remove Image"):
                run_ssh_command(f"docker rmi {image_name}", f"Remove {image_name}")
        
        with col2:
            st.subheader("Container Management")
            container_id = st.text_input("Container ID/Name")
            
            if st.button("List Containers"):
                output = run_ssh_command("docker ps -a", "List Containers")
                st.text(output)
            
            if st.button("Stop Container") and container_id:
                run_ssh_command(f"docker stop {container_id}", f"Stop {container_id}")
            
            if st.button("Remove Container") and container_id:
                run_ssh_command(f"docker rm {container_id}", f"Remove {container_id}")
    else:
        docker_client = get_docker_client()
        if docker_client is None:
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Image Management")
            image_name = st.text_input("Image Name", "nginx:latest")
            
            if st.button("Pull Image"):
                with st.spinner(f"Pulling {image_name}..."):
                    try:
                        docker_client.images.pull(image_name)
                        st.success(f"Image {image_name} pulled successfully!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            if st.button("Remove Image"):
                try:
                    docker_client.images.remove(image_name)
                    st.success(f"Image {image_name} removed!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            
            if st.button("List Images"):
                try:
                    images = docker_client.images.list()
                    if images:
                        image_data = []
                        for img in images:
                            tags = img.tags[0] if img.tags else "N/A"
                            image_data.append({
                                "ID": img.short_id,
                                "Tags": tags,
                                "Size": f"{img.attrs['Size']/1000000:.1f} MB"
                            })
                        st.dataframe(pd.DataFrame(image_data))
                    else:
                        st.info("No Docker images found")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        with col2:
            st.subheader("Container Management")
            container_id = st.text_input("Container ID/Name")
            
            if st.button("List Containers"):
                try:
                    containers = docker_client.containers.list(all=True)
                    if containers:
                        container_data = []
                        for c in containers:
                            container_data.append({
                                "ID": c.short_id,
                                "Name": c.name,
                                "Status": c.status,
                                "Image": c.image.tags[0] if c.image.tags else "N/A"
                            })
                        st.dataframe(pd.DataFrame(container_data))
                    else:
                        st.info("No Docker containers found")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            
            if st.button("Stop Container") and container_id:
                try:
                    container = docker_client.containers.get(container_id)
                    container.stop()
                    st.success(f"Container {container_id} stopped!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            
            if st.button("Remove Container") and container_id:
                try:
                    container = docker_client.containers.get(container_id)
                    container.remove()
                    st.success(f"Container {container_id} removed!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# --- Kubernetes Management ---
def kubernetes_tasks():
    st.header("Kubernetes Management")
    
    if st.session_state.ssh_host:
        st.info(f"Running Kubernetes commands on remote host: {st.session_state.ssh_host}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Pod Operations")
            pod_name = st.text_input("Pod Name", "my-pod")
            pod_image = st.text_input("Container Image", "nginx:latest")
            
            if st.button("Launch Pod"):
                run_ssh_command(f"kubectl run {pod_name} --image={pod_image}", f"Launch pod {pod_name}")
            
            if st.button("Delete Pod") and pod_name:
                run_ssh_command(f"kubectl delete pod {pod_name}", f"Delete pod {pod_name}")
        
        with col2:
            st.subheader("Cluster Monitoring")
            if st.button("List Running Pods"):
                output = run_ssh_command("kubectl get pods", "List Pods")
                st.text(output)
            
            if st.button("List Services"):
                output = run_ssh_command("kubectl get services", "List Services")
                st.text(output)
    else:
        try:
            config.load_kube_config()
            k8s_client = client.CoreV1Api()
            apps_v1 = client.AppsV1Api()
            st.success("✅ Kubernetes connection established")
        except Exception as e:
            st.error(f"❌ Kubernetes not available: {str(e)}")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Pod Operations")
            pod_name = st.text_input("Pod Name", "my-pod")
            pod_image = st.text_input("Container Image", "nginx:latest")
            
            if st.button("Launch Pod"):
                pod_manifest = {
                    "apiVersion": "v1",
                    "kind": "Pod",
                    "metadata": {"name": pod_name},
                    "spec": {
                        "containers": [{
                            "name": "main-container",
                            "image": pod_image
                        }]
                    }
                }
                try:
                    k8s_client.create_namespaced_pod(
                        body=pod_manifest, 
                        namespace="default"
                    )
                    st.success(f"Pod {pod_name} created successfully!")
                except Exception as e:
                    st.error(f"K8s Error: {str(e)}")
            
            if st.button("Delete Pod") and pod_name:
                try:
                    k8s_client.delete_namespaced_pod(
                        name=pod_name,
                        namespace="default"
                    )
                    st.success(f"Pod {pod_name} deleted!")
                except Exception as e:
                    st.error(f"K8s Error: {str(e)}")
        
        with col2:
            st.subheader("Cluster Monitoring")
            if st.button("List Running Pods"):
                try:
                    pods = k8s_client.list_namespaced_pod(namespace="default")
                    if pods.items:
                        pod_data = []
                        for pod in pods.items:
                            pod_data.append({
                                "Name": pod.metadata.name,
                                "Status": pod.status.phase,
                                "Node": pod.spec.node_name,
                                "IP": pod.status.pod_ip
                            })
                        st.dataframe(pd.DataFrame(pod_data))
                    else:
                        st.info("No pods found in default namespace")
                except Exception as e:
                    st.error(f"K8s Error: {str(e)}")
            
            if st.button("List Services"):
                try:
                    services = k8s_client.list_namespaced_service(namespace="default")
                    if services.items:
                        service_data = []
                        for svc in services.items:
                            service_data.append({
                                "Name": svc.metadata.name,
                                "Type": svc.spec.type,
                                "Cluster IP": svc.spec.cluster_ip,
                                "Ports": str(svc.spec.ports)
                            })
                        st.dataframe(pd.DataFrame(service_data))
                    else:
                        st.info("No services found in default namespace")
                except Exception as e:
                    st.error(f"K8s Error: {str(e)}")

# --- Terraform Automation ---
def terraform_tasks():
    st.header("Terraform Automation")
    
    if st.session_state.ssh_host:
        st.info(f"Running Terraform commands on remote host: {st.session_state.ssh_host}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Terraform Operations")
            if st.button("Terraform Init"):
                run_ssh_command_stream("terraform init", "Terraform Init")
            
            if st.button("Terraform Plan"):
                run_ssh_command_stream("terraform plan", "Terraform Plan")
            
            if st.button("Terraform Apply", type="primary"):
                run_ssh_command_stream("terraform apply -auto-approve", "Terraform Apply")
        
        with col2:
            st.subheader("Infrastructure Management")
            if st.button("Show Terraform State"):
                output = run_ssh_command("terraform show", "Terraform State")
                st.text(output)
            
            if st.button("Terraform Destroy", type="secondary"):
                run_ssh_command_stream("terraform destroy -auto-approve", "Terraform Destroy")
    else:
        # Directory for Terraform files
        tf_dir = "terraform"
        os.makedirs(tf_dir, exist_ok=True)
        
        # Sample Terraform configuration
        main_tf = """
        provider "aws" {
          region = "us-east-1"
        }

        resource "aws_instance" "example" {
          ami           = "ami-0c55b159cbfafe1f0"
          instance_type = "t2.micro"
          tags = {
            Name = "ExampleInstance"
          }
        }
        """
        
        # Initialize Terraform if not already
        if not os.path.exists(os.path.join(tf_dir, "main.tf")):
            with open(os.path.join(tf_dir, "main.tf"), "w") as f:
                f.write(main_tf)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Terraform Operations")
            if st.button("Terraform Init"):
                output = run_command("cd terraform && terraform init", "Terraform Init")
                st.text_area("Output", output, height=200)
            
            if st.button("Terraform Plan"):
                output = run_command("cd terraform && terraform plan", "Terraform Plan")
                st.text_area("Output", output, height=300)
            
            if st.button("Terraform Apply", type="primary"):
                output = run_command("cd terraform && terraform apply -auto-approve", "Terraform Apply")
                st.text_area("Output", output, height=300)
        
        with col2:
            st.subheader("Infrastructure Management")
            if st.button("Show Terraform State"):
                output = run_command("cd terraform && terraform show", "Terraform State")
                st.text_area("State", output, height=300)
            
            if st.button("Terraform Destroy", type="secondary"):
                output = run_command("cd terraform && terraform destroy -auto-approve", "Terraform Destroy")
                st.text_area("Output", output, height=300)

# --- Generative AI Tasks ---
def generative_ai_tasks():
    st.header("Generative AI Integration")
    
    # Gemini API Key
    st.session_state.gemini_api_key = st.text_input("Gemini API Key", 
                                                  type="password",
                                                  value=st.session_state.gemini_api_key)
    
    if not st.session_state.gemini_api_key:
        st.warning("Enter Gemini API key to continue")
        return
    
    genai.configure(api_key=st.session_state.gemini_api_key)
    
    tab1, tab2 = st.tabs(["Voice Command Execution", "Flask Deployment Assistant"])
    
    with tab1:
        st.subheader("Voice Command Execution")
        st.info("Speak a command to execute in the terminal")
        
        recognizer = sr.Recognizer()
        command_output = st.empty()
        
        def recognize_and_execute():
            with sr.Microphone() as source:
                audio = recognizer.listen(source)
                try:
                    command = recognizer.recognize_google(audio)
                    st.session_state.voice_command = command
                    
                    # Execute the command
                    if st.session_state.ssh_host:
                        result = run_ssh_command(command, "Voice Command")
                    else:
                        result = run_command(command, "Voice Command")
                    command_output.text_area("Command Result", result, height=200)
                    
                except sr.UnknownValueError:
                    st.error("Could not understand audio")
                except sr.RequestError as e:
                    st.error(f"Recognition error: {e}")
        
        if st.button("Start Listening", key="voice_command"):
            with st.spinner("Listening for command..."):
                recognize_and_execute()
        
        if 'voice_command' in st.session_state:
            st.text_area("Recognized Command", st.session_state.voice_command)
    
    with tab2:
        st.subheader("Flask Deployment Assistant")
        port = st.number_input("Port Number", 5000, min_value=1000, max_value=9999)
        app_name = st.text_input("App Name", "my_flask_app")
        
        if st.button("Generate Flask App"):
            # Generate Flask app using Gemini
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(
                    f"Create a simple Flask application with a single route that returns 'Hello World'. " 
                    f"Save it as {app_name}.py. Include instructions to run it on port {port}."
                )
                
                # Extract code from response
                code = response.text
                
                # Save to file
                with open(f"{app_name}.py", "w") as f:
                    f.write(code)
                
                st.success(f"Flask app {app_name}.py created!")
                st.code(code, language="python")
                
                # Run instructions
                run_cmd = f"python {app_name}.py --port {port}"
                st.info(f"Run your app with: {run_cmd}")
                
            except Exception as e:
                st.error(f"Gemini Error: {str(e)}")

# --- Main App Structure ---
def main():
    st.sidebar.title("Neelakshi Kaundal")
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
    st.sidebar.markdown("### Ultimate Automation Dashboard")
    
    # SSH Configuration
    with st.sidebar.expander("🔑 SSH Configuration", expanded=True):
        st.session_state.ssh_host = st.text_input("SSH Host", value=st.session_state.ssh_host)
        st.session_state.ssh_user = st.text_input("SSH Username", value=st.session_state.ssh_user)
        st.session_state.ssh_password = st.text_input("SSH Password", 
                                                    type="password",
                                                    value=st.session_state.ssh_password)
        st.session_state.ssh_key_path = st.text_input("SSH Key Path (optional)", 
                                                    value=st.session_state.ssh_key_path)
        
        if st.button("Test SSH Connection"):
            ssh = get_ssh_client()
            if ssh:
                st.success("✅ SSH connection successful!")
                ssh.close()
            else:
                st.error("❌ SSH connection failed")
    
    # API Keys Section
    with st.sidebar.expander("🔑 API Keys", expanded=True):
        st.session_state.openai_api_key = st.text_input("OpenAI API Key", 
                                                      type="password",
                                                      value=st.session_state.openai_api_key)
        st.session_state.gemini_api_key = st.text_input("Gemini API Key", 
                                                      type="password",
                                                      value=st.session_state.gemini_api_key)
        st.session_state.google_api_key = st.text_input("Google API Key", 
                                                      type="password",
                                                      value=st.session_state.google_api_key)
        st.session_state.google_cse_id = st.text_input("Google CSE ID", 
                                                     value=st.session_state.google_cse_id)
        st.session_state.aws_access_key = st.text_input("AWS Access Key", 
                                                      type="password",
                                                      value=st.session_state.aws_access_key)
        st.session_state.aws_secret_key = st.text_input("AWS Secret Key", 
                                                      type="password",
                                                      value=st.session_state.aws_secret_key)
    
    # Navigation
    page = st.sidebar.selectbox("Select Category", [
        "Git Tasks", 
        "Machine Learning Tasks", 
        "Web Development Tasks",
        "AWS EC2 Management",
        "Docker Management",
        "Kubernetes Management",
        "Terraform Automation",
        "Generative AI Tasks"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.info("All tasks can be executed directly from this dashboard")
    
    # Page Header
    st.title(f"Automation Dashboard: {page}")
    st.markdown("---")
    
    # Page Routing
    if page == "Git Tasks":
        git_tasks()
    elif page == "Machine Learning Tasks":
        ml_tasks()
    elif page == "Web Development Tasks":
        webdev_tasks()
    elif page == "AWS EC2 Management":
        aws_ec2_tasks()
    elif page == "Docker Management":
        docker_tasks()
    elif page == "Kubernetes Management":
        kubernetes_tasks()
    elif page == "Terraform Automation":
        terraform_tasks()
    elif page == "Generative AI Tasks":
        generative_ai_tasks()

if __name__ == "__main__":
    main()
