# SynthMining-ML-Learning

## Description
This project demonstrates the construction and training of a simple Convolutional Neural Network (CNN) for image classification using PyTorch. The goal is to build a model capable of classifying handwritten digits from the MNIST dataset. The MNIST dataset consists of 28x28 grayscale images of digits (0-9). This should help you get familiarized with the main concepts of machine learning as well as the basics of working with pytorch.

## Getting Started

Follow these instructions to set up and run the project on your local machine. Once you have set up your environment, navigate to the `pytorch_intro` directory where you will be able to develop the CNN classification model and training pipeline. If you would like more practice before beginning, implement the linear regression and feed forward networks first. These will be simpler and will help you gain practice working with pytorch before you delve into the CNN.

### Prerequisites

- Python 3.11

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/your-repository.git
    ```

2. Navigate to the project directory:

    ```bash
    cd your-repository
    ```

3. Create a virtual environment:

    ```bash
    python3.11 -m venv venv
    ```

    For Windows users:

    ```bash
    python -m venv venv
    ```

4. Activate the virtual environment:

    On Linux/macOS:

    ```bash
    source venv/bin/activate
    ```

    On Windows:

    ```bash
    .\venv\Scripts\activate
    ```

5. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

6. Install Jupyter Notebook (optional):

    ```bash
    pip install jupyter
    ```

    If you're planning to use Jupyter Notebook.

7. Create a Jupyter kernel (optional):

    ```bash
    python -m ipykernel install --user --name=venv --display-name="Your Kernel Name"
    ```

    Replace "venv" with your virtual environment name and "Your Kernel Name" with the desired display name.

8. Deactivate the virtual environment when you're done:

    ```bash
    deactivate
    ```