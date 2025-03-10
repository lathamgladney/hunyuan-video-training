# Hunyuan Video Generation Pipeline

This project extends the training pipeline from [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) repository, adding Modal cloud integration and enhanced video generation capabilities.

## 🚀 Key Features

- **Cloud-based Training:** Train Hunyuan video models in the cloud using Modal's infrastructure
- **ComfyUI Integration:** Generate videos using trained models with an intuitive UI
- **Hugging Face Integration:** Upload and download models/checkpoints to/from Hugging Face
- **Tensorboard Support:** Track and visualize training progress
- **Customizable Configuration:** Adjust training and inference parameters through a user-friendly interface
- **GPU Acceleration:** Optimized for high-performance GPU training and inference

## ⚙️ System Requirements

### Local Environment
You need the following on your local machine:

- Python 3.10+

### Required Python Packages
The following Python packages are required locally:

```
gradio==5.20.0
fastapi==0.115.11
pydantic==2.10.6
starlette==0.46.0
modal==0.67.43
requests==2.32.3
toml==0.10.2
tensorboard==2.19.0
huggingface_hub==0.29.0
```

These dependencies are listed in the `requirements.txt` file for easy installation.

## 🛠 Installation Guide

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AINxtGen/hunyuan-video-training.git
   cd hunyuan-video-training
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # For Windows
   python -m venv hytraining
   .\hytraining\Scripts\activate

   # For macOS/Linux
   python -m venv hytraining
   source hytraining/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Modal account:**
   - Create an account on [Modal](https://modal.com/)
   - Authenticate with Modal: `modal token new` or you can go to https://modal.com/settings/tokens and create a New Token

5. **Set up Hugging Face token:**
   - Create a Hugging Face account and generate an access token
   - Store your token in Modal:
     ```bash
     modal secret create huggingface-token HF_TOKEN="your_hf_token_here"
     ```

6. **Deploy the application:**
   ```bash
   modal deploy main.py
   ```

7. **Launch the UI:**
   ```bash
   python ui_setting.py
   ```

8. **Login to Hugging Face:**
   ```bash
   huggingface-cli login
   ```
   This step is required for downloading files from Hugging Face repositories.

## 📝 Usage

1. Open the UI by running `python ui_setting.py`
2. Place your dataset folder in the same directory as this script
3. Configure your training parameters
4. Start training by clicking "Start Training" button
5. Monitor training progress using Tensorboard
6. Generate videos using the trained models via ComfyUI API
