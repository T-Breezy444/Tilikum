
# My First LLM

This project represents my journey into building a large language model (LLM) from scratch. Since starting graduate school, I've developed a deep interest in large language models, and this project is an exciting step in expanding that knowledge further. A significant part of my learning has come from the [LLM From Scratch Course](https://www.youtube.com/watch?v=UU1WVnMk4E8), which I highly recommend to anyone interested in understanding the intricacies of LLMs.

## Project Overview

This repository contains the code for training a custom transformer-based language model using PyTorch. The model is designed to process text sequences, generate predictions, and learn from large datasets. Key features include:

- **Multi-Head Self-Attention**: Implements the attention mechanism with multiple heads for parallelization.
- **FeedForward Network**: A simple yet efficient two-layer network with a ReLU activation.
- **Transformer Blocks**: The model is built with several transformer blocks that handle communication between tokens followed by computation.
- **Mixed Precision Training**: Utilizes PyTorch AMP (Automatic Mixed Precision) for efficient training on CUDA.

## How to Use

### Prerequisites

Make sure you have the following installed:

- Python 3.8+
- PyTorch
- Hugging Face `transformers` library
- CUDA-compatible GPU (optional, but recommended)

### Running the Code

To train the model, adjust any of the necessary parameters in the script and run the following command:

```bash
python train_model.py
```

You can customize the training configuration, such as `batch_size`, `learning_rate`, and other hyperparameters in the script.

### File Structure

- `train_model.py`: Main script for training the LLM. This includes data loading, model initialization, and the training loop.
- `RedditVocab.txt`: The vocabulary file generated from the training corpus.
- `best.pkl`: A sample checkpoint of the best-performing model.
  
Feel free to tweak the script to fit your dataset or desired configuration.

### Model Saving and Loading

During training, the model is automatically saved when an improvement in validation loss is detected. You can load a saved model by uncommenting the provided section in the script and specifying the appropriate path to the model file.

## Future
Soon I plan on fine-tuning the model to a code data set and to be able to output better responses. I have large ambitions of helping the common man build an LLM to suit his needs on limited hardware. I believe AI is for all and this is my first step in helping with that goal. I know I am not the first to build a smaller model and certainly not the last but it is the start of a long journey. 

I also plan on writing a few papers and self-publishing unless any lab has an opening and wants to reach out! if so look below, if not you'll see my paper someday some way!

## Contact

If you have any questions or feedback, feel free to reach out to me on [LinkedIn](https://www.linkedin.com/in/tyler-blalock/).

I hope to continue refining and expanding this model and look forward to collaborating with others in the LLM space.

Have a great day reader!
