import math
import unicodedata
import numpy as np
import mmap
import os
import sys
from pathlib import Path

def replace_control_characters(s: str) -> str:
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)
        else:
            chars.append(f"\\u{ord(ch):04x}")
    return "".join(chars)

def render_token(t: bytes) -> str:
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

class EntropyTokenizer:
    def __init__(self, global_threshold=None, relative_threshold=None, window_size=1000, chunk_size=1024*1024):
        self.global_threshold = global_threshold
        self.relative_threshold = relative_threshold
        self.vocab = {}  # idx -> bytes
        self.special_tokens = {}  # str -> int
        self.window_size = window_size  # Size of sliding window for entropy calculation
        self.chunk_size = chunk_size  # Size of chunks for processing mmap

    def train(self, text_file, verbose=False):
        file_size = os.path.getsize(text_file)

        # First pass: Build transition matrix
        print("\rBuilding transition matrix...", end="", flush=True)
        transition_matrix = np.zeros((256, 256), dtype=np.float32)
        byte_counts = np.zeros(256, dtype=np.int32)

        with open(text_file, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                # Process file in chunks to build transition matrix
                for chunk_start in range(0, file_size, self.chunk_size):
                    chunk_end = min(chunk_start + self.chunk_size, file_size)
                    chunk = mm[chunk_start:chunk_end]

                    # Count transitions and bytes in this chunk
                    for i in range(len(chunk) - 1):
                        curr_byte = chunk[i]
                        next_byte = chunk[i + 1]
                        transition_matrix[curr_byte, next_byte] += 1
                        byte_counts[curr_byte] += 1

                    # Count the last byte of the chunk
                    if chunk:
                        byte_counts[chunk[-1]] += 1

                    print(f"\rProcessed {chunk_end}/{file_size} bytes for transition matrix",
                          end="", flush=True)

        # Calculate probabilities
        for i in range(256):
            if byte_counts[i] > 0:
                transition_matrix[i] /= byte_counts[i]

        # Second pass: Calculate entropies using sliding windows
        print("\nCalculating entropies...", end="", flush=True)
        entropies = []

        with open(text_file, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                # Process file in chunks for entropy calculation
                for chunk_start in range(0, file_size - 1, self.chunk_size):
                    chunk_end = min(chunk_start + self.chunk_size, file_size - 1)
                    chunk = mm[chunk_start:chunk_end]

                    # Calculate entropies for this chunk
                    chunk_entropies = self._calculate_entropies_windowed(chunk, transition_matrix)
                    entropies.extend(chunk_entropies)

                    print(f"\rCalculated entropies for {chunk_end}/{file_size} bytes",
                          end="", flush=True)

        # Find boundaries
        print("\nFinding boundaries...", end="", flush=True)
        boundaries = self._identify_boundaries(entropies)

        # Third pass: Extract chunks using boundaries
        print("\nExtracting text chunks...", end="", flush=True)
        chunks = []
        current_pos = 0

        with open(text_file, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                for boundary in boundaries:
                    chunk = mm[current_pos:boundary]
                    try:
                        # Try to decode as UTF-8
                        chunk_text = chunk.decode('utf-8')
                        chunks.append(chunk_text)
                    except UnicodeDecodeError:
                        # If chunk breaks UTF-8 character, adjust boundary
                        for i in range(4):  # UTF-8 char can be up to 4 bytes
                            try:
                                chunk = mm[current_pos:boundary-i]
                                chunk_text = chunk.decode('utf-8')
                                chunks.append(chunk_text)
                                boundary -= i
                                break
                            except UnicodeDecodeError:
                                continue
                    current_pos = boundary

                # Add final chunk if needed
                if current_pos < file_size:
                    final_chunk = mm[current_pos:file_size]
                    try:
                        chunks.append(final_chunk.decode('utf-8'))
                    except UnicodeDecodeError:
                        # Handle potential truncated UTF-8 at end of file
                        chunks.append(final_chunk[:-1].decode('utf-8', errors='ignore'))

        print(f"\rIdentified {len(chunks)} text chunks", end="", flush=True)

        # Build vocabulary from chunks
        self._build_vocab(chunks, verbose)

    def _calculate_entropies_windowed(self, chunk, transition_matrix):
        entropies = []

        # Process chunk in windows for better cache utilization
        for i in range(1, len(chunk)):
            prev_byte = chunk[i-1]
            curr_byte = chunk[i]
            prob = transition_matrix[prev_byte, curr_byte]
            if prob > 0:
                entropy = -prob * math.log2(prob)
                entropies.append(entropy)
            else:
                entropies.append(0)

        return entropies

    def _identify_boundaries(self, entropies):
        boundaries = set()

        # Convert to numpy array for faster operations
        entropies_array = np.array(entropies)

        if self.global_threshold is not None:
            # Vectorized comparison
            high_entropy_indices = np.where(entropies_array > self.global_threshold)[0]
            boundaries.update(high_entropy_indices + 1)

        if self.relative_threshold is not None and len(entropies_array) > 1:
            # Vectorized difference calculation
            entropy_differences = np.diff(entropies_array)
            significant_changes = np.where(entropy_differences > self.relative_threshold)[0]
            boundaries.update(significant_changes + 1)

        return sorted(list(boundaries))

    def _build_vocab(self, chunks, verbose=False):
        # Start with basic byte vocabulary
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        next_id = 256
        
        # Create reverse mapping for checking duplicates
        token_to_id = {bytes([idx]): idx for idx in range(256)}
        skipped_tokens = []  # List to track skipped tokens
        
        total_chunks_processed = 0
        
        for chunk in chunks:
            if not chunk:  # Skip empty chunks
                continue
                
            chunk_bytes = chunk.encode('utf-8')
            total_chunks_processed += 1
            
            # Check if this token content already exists
            if chunk_bytes in token_to_id:
                skipped_tokens.append((chunk, token_to_id[chunk_bytes]))  # Store skipped token and its existing ID
                sys.stdout.write(f"\rProcessed chunks: {total_chunks_processed}, Unique tokens: {len(token_to_id) - 256}")
                sys.stdout.flush()
                continue
                
            # Add new token
            self.vocab[next_id] = chunk_bytes
            token_to_id[chunk_bytes] = next_id
            
            if verbose:
                print(f"\nAdded token {next_id}: \"{repr(chunk)[1:-1]}\"")
            next_id += 1
            sys.stdout.write(f"\rProcessed chunks: {total_chunks_processed}, Unique tokens: {len(token_to_id) - 256}")
            sys.stdout.flush()
        
        print()  # Final newline after progress indicator

    def save(self, file_prefix):
        # Create reverse mapping to check for duplicates
        content_to_id = {}
        filtered_vocab = {}
        
        # Filter out duplicates while preserving the lowest ID for each unique content
        for idx, token in sorted(self.vocab.items()):  # Sort by ID to prefer lower IDs
            token_content = bytes(token)  # Ensure we're comparing bytes
            if token_content not in content_to_id:
                content_to_id[token_content] = idx
                filtered_vocab[idx] = token

        # Save the model file
        model_file = Path(file_prefix).stem + ".model"
        with open(model_file, 'w', encoding='utf-8') as f:
            f.write("entropy_tokenizer v1\n")
            
            # Write thresholds
            f.write(f"{self.global_threshold if self.global_threshold else 'None'}\n")
            f.write(f"{self.relative_threshold if self.relative_threshold else 'None'}\n")
            
            # Write special tokens after checking for duplicates
            unique_special_tokens = {}
            for special, idx in self.special_tokens.items():
                token_bytes = special.encode('utf-8')
                if token_bytes not in content_to_id:
                    unique_special_tokens[special] = idx
                    content_to_id[token_bytes] = idx
            
            f.write(f"{len(unique_special_tokens)}\n")
            for special, idx in unique_special_tokens.items():
                f.write(f"{special} {idx}\n")
            
            # Write vocabulary entries (excluding basic bytes)
            for idx, token in filtered_vocab.items():
                if idx >= 256:  # Only write tokens beyond basic bytes
                    hex_bytes = ' '.join(f"{b:02x}" for b in token)
                    f.write(f"{idx} {hex_bytes}\n")

        # Save human-readable vocabulary file
        vocab_file = Path(file_prefix).stem + ".vocab"
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for idx, token in filtered_vocab.items():
                try:
                    # Decode the token bytes to string and use repr() to properly escape special characters
                    token_str = token.decode('utf-8', errors='replace')
                    # Use repr() to get escaped string and remove the outer quotes
                    escaped_str = repr(token_str)[1:-1]
                    f.write(f"[{escaped_str}] {idx}\n")
                except UnicodeDecodeError:
                    # Fallback for any bytes that can't be decoded
                    hex_repr = ' '.join(f"\\x{b:02x}" for b in token)
                    f.write(f"[{hex_repr}] {idx}\n")

    def load(self, model_file):
        assert model_file.endswith(".model")
        
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        with open(model_file, 'r', encoding='utf-8') as f:
            # Read version
            version = f.readline().strip()
            assert version == "entropy_tokenizer v1"
            
            # Read thresholds
            global_thresh = f.readline().strip()
            self.global_threshold = float(global_thresh) if global_thresh != 'None' else None
            
            relative_thresh = f.readline().strip()
            self.relative_threshold = float(relative_thresh) if relative_thresh != 'None' else None
            
            # Read special tokens
            num_special = int(f.readline().strip())
            self.special_tokens = {}
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                self.special_tokens[special] = int(special_idx)
            
            # Read vocabulary entries
            for line in f:
                parts = line.strip().split()
                idx = int(parts[0])
                token_bytes = bytes.fromhex(''.join(parts[1:]))
                self.vocab[idx] = token_bytes

    def encode(self, text):
        """Encodes text into token IDs"""
        if not text:
            return []

        # For encoding single texts, we'll use the simpler in-memory approach
        # since mmap would be overkill for small strings
        encoded_text = text.encode('utf-8')

        # Create transition matrix for encoding
        transition_matrix = np.zeros((256, 256), dtype=np.float32)
        byte_counts = np.zeros(256, dtype=np.int32)

        for i in range(len(encoded_text) - 1):
            curr_byte = encoded_text[i]
            next_byte = encoded_text[i + 1]
            transition_matrix[curr_byte, next_byte] += 1
            byte_counts[curr_byte] += 1

        for i in range(256):
            if byte_counts[i] > 0:
                transition_matrix[i] /= byte_counts[i]

        entropies = self._calculate_entropies_windowed(encoded_text, transition_matrix)
        boundaries = self._identify_boundaries(entropies)
        chunks = []
        start = 0
        for boundary in boundaries:
            chunk = encoded_text[start:boundary].decode('utf-8', errors='ignore')
            chunks.append(chunk)
            start = boundary
        if start < len(encoded_text):
            chunks.append(encoded_text[start:].decode('utf-8', errors='ignore'))

        ids = []
        for chunk in chunks:
            if chunk in self.special_tokens:
                ids.append(self.special_tokens[chunk])
                continue

            chunk_bytes = chunk.encode('utf-8')
            # Try to find the chunk in vocabulary
            found = False
            for idx, token in self.vocab.items():
                if chunk_bytes == token:
                    ids.append(idx)
                    found = True
                    break

            # If not found, encode as individual bytes
            if not found:
                ids.extend(list(chunk_bytes))

        return ids

    def decode(self, ids):
        """Decodes token IDs back into text"""
        parts = []
        for idx in ids:
            if idx in self.vocab:
                parts.append(self.vocab[idx])
            elif isinstance(idx, str) and idx in self.special_tokens:
                parts.append(self.special_tokens[idx].encode('utf-8'))
            elif isinstance(idx, int) and idx in self.special_tokens:
                # Assuming special tokens are stored as strings
                # If they were stored as bytes, this would need adjustment
                parts.append(self.special_tokens[idx].encode('utf-8'))
            else:
                raise ValueError(f"Invalid token ID: {idx}")

        return b''.join(parts).decode('utf-8', errors='replace')
