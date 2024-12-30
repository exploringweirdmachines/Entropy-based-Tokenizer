# Entropy-Based Tokenizer

"Hey, look! A regex free tokenizer!"  

A tokenizer that uses entropy calculations to identify natural boundaries in text and create tokens. This tokenizer is particularly effective for processing large text files as it uses memory mapping (mmap) and chunked processing.<br>
<br>
Based on Andrej Karpathy's minBPE tokenizer and inspired by META's 'Byte Latent Transformer: Patches Scale Better Than Tokens' paper.

## Features

- **Entropy-based token discovery**: Uses byte-level entropy calculations to find natural word and subword boundaries
- **Memory efficient**: Uses memory mapping (mmap) to process large files without loading them entirely into memory
- **Flexible thresholds**: Two configurable thresholds that can be used independently or in combination:
  - Global threshold: Identifies boundaries based on absolute entropy values
  - Relative threshold: Identifies boundaries where there are significant jumps in entropy between consecutive positions
- **Progress monitoring**: Real-time progress updates during processing
- **UTF-8 support**: Properly handles UTF-8 encoded text and character boundaries
- **Model persistence**: Save and load trained tokenizer models

## Installation

### Requirements

```bash
numpy
```

### Setup

Clone this repository:

```bash
git clone https://github.com/exploringweirdmachines/Entropy-based-Tokenizer.git
```

## Usage

### Training and Inference

The tokenizer can be used in two modes: training and inference. Both modes are accessible through the command line interface.

#### Training Mode
```bash
usage: train_entropy_tokenizer.py train [-h] [--global_threshold GLOBAL_THRESHOLD] [--relative_threshold RELATIVE_THRESHOLD] [--window_size WINDOW_SIZE] [--chunk_size CHUNK_SIZE] [train_file] model_name

positional arguments:
  train_file            Path to the text file to train on (default: shakespeare.txt).
  model_name            Name of the model to save.

options:
  -h, --help            show this help message and exit
  --global_threshold GLOBAL_THRESHOLD
                        Global threshold for the tokenizer.
  --relative_threshold RELATIVE_THRESHOLD
                        Relative threshold for the tokenizer.
  --window_size WINDOW_SIZE
                        Window size for the tokenizer.
  --chunk_size CHUNK_SIZE
                        Chunk size for the tokenizer (default: 1MB).
```

```bash
python train_entropy_tokenizer.py train shakespeare.txt

Building transition matrix...
Processed 1115394/1115394 bytes for transition matrix
Calculating entropies...
Finding boundaries...
Identified 656231 text chunks
Added token 256: Fir
Processed chunks: 1, Unique tokens: 1
Added token 257: s
Processed chunks: 2, Unique tokens: 2
Added token 258: t 
Processed chunks: 3, Unique tokens: 3
Added token 259: C
Processed chunks: 4, Unique tokens: 4
Added token 260: iti
Processed chunks: 5, Unique tokens: 5
Added token 261: z
Processed chunks: 6, Unique tokens: 6
...
Processed chunks: 656060, Unique tokens: 7619
Added token 7875: surel
Processed chunks: 656152, Unique tokens: 7620
Added token 7876: din
Processed chunks: 656171, Unique tokens: 7621
Added token 7877: st a
Processed chunks: 656206, Unique tokens: 7622
Added token 7878: p--di
Processed chunks: 656230, Unique tokens: 7623
Added token 7879: king.

Processed chunks: 656231, Unique tokens: 7624
Training took 32.69 seconds
```

#### Inference Mode

```bash
usage: train_entropy_tokenizer.py inference [-h] [--input_text INPUT_TEXT] [--input_file INPUT_FILE] [model_name]

positional arguments:
  model_name            Path to the tokenizer model to load (default: shakespear.model).

options:
  -h, --help            show this help message and exit
  --input_text INPUT_TEXT
                        Text to encode and decode (default: "Three Laws of Robotics").
  --input_file INPUT_FILE
                        Path to file that has the input text to encode and decode.
```

```bash
python train_entropy_tokenizer.py inference
Loaded text: "Isaac Asimov's "Three Laws of Robotics"
1.A robot may not injure a human being or, through inaction, allow a human being to come to harm.
2.A robot must obey orders given it by human beings except where such orders would conflict with the First Law.
3.A robot must protect its own existence as long as such protection does not conflict with the First or Second Law.
"
Encoded text to tokens: [73, 2896, 97, 99, 32, 65, 115, 105, 492, 118, 39, 115, 32, 34, 84, 6449, 101, 6717, 6322, 115, 526, 102, 3367, 111, 98, 1062, 105, 99, 115, 34, 10, 49, 46, 65, 32, 114, 111, 98, 6316, 109, 786, 32, 110, 6316, 105, 110, 106, 117, 114, 101, 32, 1133, 104, 117, 109, 97, 110, 32, 98, 101, 105, 110, 103, 32, 111, 114, 44, 32, 6114, 114, 111, 117, 103, 104, 32, 105, 110, 97, 99, 901, 2736, 295, 108, 108, 111, 313, 1133, 104, 117, 109, 97, 110, 32, 98, 101, 105, 110, 103, 32, 116, 111, 32, 99, 111, 109, 101, 32, 116, 111, 32, 104, 97, 114, 109, 822, 50, 46, 65, 32, 114, 111, 98, 6316, 2878, 115, 258, 111, 98, 747, 32, 111, 114, 100, 101, 6151, 32, 465, 298, 110, 32, 6136, 1008, 32, 104, 117, 109, 97, 110, 32, 98, 101, 105, 715, 115, 32, 101, 120, 99, 101, 112, 258, 119, 6105, 114, 101, 32, 115, 996, 104, 32, 111, 114, 100, 101, 6151, 32, 659, 117, 108, 100, 32, 99, 353, 102, 108, 105, 99, 258, 1907, 104, 32, 116, 104, 101, 32, 70, 105, 6151, 116, 32, 76, 97, 119, 46, 10, 51, 46, 65, 32, 114, 111, 98, 6316, 2878, 115, 258, 112, 114, 2202, 99, 258, 381, 115, 526, 119, 110, 32, 101, 1758, 115, 6561, 99, 101, 32, 97, 115, 32, 108, 111, 110, 103, 32, 97, 115, 32, 115, 996, 104, 32, 112, 114, 2202, 99, 901, 111, 110, 32, 453, 101, 115, 32, 110, 6316, 99, 353, 102, 108, 105, 99, 258, 1907, 104, 32, 116, 104, 101, 32, 70, 105, 6151, 116, 32, 111, 114, 32, 83, 101, 99, 353, 100, 6717, 97, 119, 46, 10]

Decoded tokens back to text: "Isaac Asimov's "Three Laws of Robotics"
1.A robot may not injure a human being or, through inaction, allow a human being to come to harm.
2.A robot must obey orders given it by human beings except where such orders would conflict with the First Law.
3.A robot must protect its own existence as long as such protection does not conflict with the First or Second Law.
```

### Programmatic Usage

```python
from entropy import EntropyTokenizer

# Initialize the tokenizer with default settings
tokenizer = EntropyTokenizer(
    global_threshold=0.5,    # Default threshold for absolute entropy values
    relative_threshold=0.03, # Default threshold for entropy changes
    window_size=1000,       # Default window size
    chunk_size=1024*1024    # Default chunk size (1MB)
)

# Train on text file (defaults to shakespeare.txt)
tokenizer.train("input.txt", verbose=True)

# Save the trained model (will create .model and .vocab files)
tokenizer.save("my_model")

# Load a model
loaded_tokenizer = EntropyTokenizer()
loaded_tokenizer.load("my_model.model")  # Defaults to shakespeare.model

# Encode text
text = "Example text to encode"
tokens = loaded_tokenizer.encode(text)

# Decode tokens
decoded_text = loaded_tokenizer.decode(tokens)
```

### Advanced Configuration

The tokenizer supports several configuration parameters that can be adjusted based on your needs:

```python
tokenizer = EntropyTokenizer(
    global_threshold=0.5,    # Controls absolute entropy threshold for token boundaries
    relative_threshold=0.03, # Controls relative entropy change threshold
    window_size=1000,       # Size of sliding window for entropy calculation
    chunk_size=1024*1024    # Size of chunks for memory-mapped processing (1MB)
)
```

- `global_threshold`: Higher values create fewer tokens (default: 0.5)
- `relative_threshold`: Higher values make the tokenizer less sensitive to entropy changes (default: 0.03)
- `window_size`: Controls the context window for entropy calculations (default: 1000)
- `chunk_size`: Controls memory usage during processing (default: 1MB)

## How It Works

1. **Transition Matrix Building**
   - Creates a 256x256 matrix representing byte transition probabilities
   - Processes the input file in chunks using memory mapping
   - Counts byte occurrences and transitions

2. **Entropy Calculation**
   - Uses the transition matrix to calculate byte-level entropies
   - Identifies potential token boundaries based on:
     - High absolute entropy values (global threshold)
     - Significant changes in entropy (relative threshold)

3. **Token Extraction**
   - Extracts text chunks based on identified boundaries
   - Ensures proper UTF-8 character boundary handling
   - Builds a vocabulary of unique tokens

4. **Model Storage**
   - Saves the model in two formats:
     - `.model`: Binary model file for loading
     - `.vocab`: Human-readable vocabulary file

## Model File Format

The `.model` file contains:
- Version identifier
- Global and relative thresholds
- Special tokens (if any)
- Vocabulary entries (token index and byte representation)

The `.vocab` file shows:
- Human-readable representation of each token
- Token indices
- Special characters properly escaped

## Performance Considerations

- Memory usage remains relatively constant regardless of input file size
- Processing speed scales linearly with file size
- Optimal chunk_size depends on available system memory
- Adjust window_size to balance between precision and processing speed

## Todo

- Limit for the number of tokens to be generated
- Add special tokens capability

## License

Apache License 2.0

## Citation

If you use this tokenizer in your research, please cite:

```bibtex
@software{entropy_tokenizer,
  author = {exploringweirdmachines},
  title = {Entropy-based-Tokenizer},
  year = {2024},
  url = {https://github.com/exploringweirdmachines/Entropy-based-Tokenizer}
}
```
