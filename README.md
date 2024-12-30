# Entropy-Based Tokenizer

"Hey, look! A regex free tokenizer!"<br>
A memory-efficient tokenizer that uses entropy calculations to identify natural boundaries in text and create tokens. This tokenizer is particularly effective for processing large text files as it uses memory mapping (mmap) and chunked processing.

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

### Threshold Configuration

The tokenizer supports two types of thresholds that can be used independently or together:

1. **Global Threshold**: Sets an absolute entropy value above which token boundaries are identified. Useful for finding positions of high uncertainty in the text.
   ```python
   # Using only global threshold
   tokenizer = EntropyTokenizer(global_threshold=0.5)
   ```

2. **Relative Threshold**: Identifies boundaries at positions where there's a significant jump in entropy compared to the previous position. This helps catch sudden changes in predictability.
   ```python
   # Using only relative threshold
   tokenizer = EntropyTokenizer(relative_threshold=0.03)
   ```

3. **Combined Usage**: When both thresholds are used, a position will be marked as a boundary if it satisfies either condition (high absolute entropy OR significant entropy jump).
   ```python
   # Using both thresholds
   tokenizer = EntropyTokenizer(global_threshold=0.5, relative_threshold=0.03)
   ```

### Basic Usage

```python
from entropy_tokenizer import EntropyTokenizer

# Initialize the tokenizer
tokenizer = EntropyTokenizer(
    global_threshold=0.5,    # Threshold for absolute entropy values
    relative_threshold=0.03  # Threshold for entropy changes
)

# Train on a text file
tokenizer.train("path/to/your/text.txt", verbose=True)

# Save the trained model
tokenizer.save("my_tokenizer")

# Load a saved model
loaded_tokenizer = EntropyTokenizer()
loaded_tokenizer.load("my_tokenizer.model")

# Encode text
text = "Example text to encode"
tokens = loaded_tokenizer.encode(text)

# Decode tokens
decoded_text = loaded_tokenizer.decode(tokens)
```

### Advanced Configuration

```python
tokenizer = EntropyTokenizer(
    global_threshold=0.5,       # Threshold for absolute entropy values
    relative_threshold=0.03,    # Threshold for entropy changes
    window_size=1000,          # Size of sliding window for entropy calculation
    chunk_size=1024*1024       # Size of chunks for processing mmap (1MB default)
)
```

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

## Limitations

- Currently processes single files (no directory traversal)
- Requires UTF-8 encoded input files
- Token boundaries are determined solely by entropy calculations


## License

Apache License 2.0

## Citation

If you use this tokenizer in your research, please cite:

```bibtex
@software{entropy_tokenizer,
  author = exploringweirdmachines,
  title = {Entropy-based-Tokenizer},
  year = {2024},
  url = {https://github.com/exploringweirdmachines/Entropy-based-Tokenizer}
}
```
