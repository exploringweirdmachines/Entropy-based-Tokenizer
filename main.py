# train_tokenizer.py
import time
import argparse
from .entropy import EntropyTokenizer

def train_mode(args):
    """Trains the tokenizer and saves the model."""
    t0 = time.time()

    tokenizer = EntropyTokenizer(
        global_threshold=args.global_threshold,
        relative_threshold=args.relative_threshold,
        window_size=args.window_size,
        chunk_size=args.chunk_size
    )

    # Train on a text file
    tokenizer.train(args.train_file, verbose=True)
    tokenizer.save(args.model_name)
    t1 = time.time()
    print(f"Training took {t1 - t0:.2f} seconds")

def inference_mode(args):
    """Loads a trained model and performs encoding and decoding."""
    # Load a saved model
    loaded_tokenizer = EntropyTokenizer()
    loaded_tokenizer.load(args.model_name)

    # Encode text
    text = args.input_text
    if not text and args.input_file:
        with open(args.input_file, 'r') as f:
            text = f.read()

    print(f"Loaded text: \"{text}\"")
    tokens = loaded_tokenizer.encode(text)
    print(f"Encoded text to tokens: {tokens}")
    for token in tokens:
        decoded_token = loaded_tokenizer.decode([token])
        print(f"Decoded token: \"{decoded_token}\"")
    # Decode tokens
    decoded_text = loaded_tokenizer.decode(tokens)
    print(f"Decoded tokens back to text: \"{decoded_text}\"")

def main():
    parser = argparse.ArgumentParser(description="Train and use an Entropy Tokenizer.")
    subparsers = parser.add_subparsers(title="mode", dest="mode", help="Choose either 'train' or 'inference'")

    # Training mode parser
    train_parser = subparsers.add_parser("train", help="Train a new tokenizer model.")
    train_parser.add_argument("train_file", type=str, nargs='?', default="shakespear.txt", help="Path to the text file to train on (default: shakespear.txt).")
    train_parser.add_argument("model_name", type=str, help="Name of the model to save.")
    train_parser.add_argument(
        "--global_threshold", type=float, default=0.5, help="Global threshold for the tokenizer."
    )
    train_parser.add_argument(
        "--relative_threshold", type=float, default=0.03, help="Relative threshold for the tokenizer."
    )
    train_parser.add_argument(
        "--window_size", type=int, default=1000, help="Window size for the tokenizer."
    )
    train_parser.add_argument(
        "--chunk_size", type=int, default=1024*1024, help="Chunk size for the tokenizer (default: 1MB)."
    )

    # Inference mode parser
    inference_parser = subparsers.add_parser("inference", help="Load and use a trained tokenizer model.")
    inference_parser.add_argument("model_name", type=str, nargs='?', default="shakespear.model", help="Path to the tokenizer model to load (default: shakespear.model).")
    inference_parser.add_argument(
        "--input_text", type=str, default="""Isaac Asimov's "Three Laws of Robotics"
1.A robot may not injure a human being or, through inaction, allow a human being to come to harm.
2.A robot must obey orders given it by human beings except where such orders would conflict with the First Law.
3.A robot must protect its own existence as long as such protection does not conflict with the First or Second Law.
""", help='Text to encode and decode (default: "Three Laws of Robotics").'
    )
    inference_parser.add_argument(
        "--input_file", type=str, default=None, help="Path to file that has the input text to encode and decode."
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_mode(args)
    elif args.mode == "inference":
        inference_mode(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
