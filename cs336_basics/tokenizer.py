"""
Byte Pair Encoding (BPE) tokenizer implementation.
"""
import os
import regex as re
from collections import Counter
from typing import Iterable


class BPETokenizer:
    """
    Byte Pair Encoding tokenizer.
    
    Implements BPE algorithm for subword tokenization.
    """
    
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Initialize BPE tokenizer.
        
        Args:
            vocab: Mapping from token ID to token bytes
            merges: List of BPE merges in order
            special_tokens: List of special token strings
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        # Build reverse vocab for encoding
        self.token_to_id = {token: idx for idx, token in vocab.items()}
        
        # Build merge priority dict
        self.merge_priority = {merge: i for i, merge in enumerate(merges)}
        
        # Pre-tokenization pattern (GPT-2 style)
        # Splits on whitespace and keeps punctuation separate
        self.pattern = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
            re.IGNORECASE
        )
        
        # Build special token pattern if we have special tokens
        if self.special_tokens:
            special_pattern = '|'.join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True))
            self.special_pattern = re.compile(f'({special_pattern})')
        else:
            self.special_pattern = None
    
    def _apply_merges(self, tokens: list[bytes]) -> list[bytes]:
        """Apply BPE merges to a list of byte tokens."""
        if len(tokens) <= 1:
            return tokens
        
        while True:
            # Find the pair with highest priority (lowest merge index)
            best_pair = None
            best_idx = None
            best_priority = float('inf')
            
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_priority:
                    priority = self.merge_priority[pair]
                    if priority < best_priority:
                        best_pair = pair
                        best_idx = i
                        best_priority = priority
            
            # If no mergeable pair found, we're done
            if best_pair is None:
                break
            
            # Merge the best pair
            merged = best_pair[0] + best_pair[1]
            new_tokens = tokens[:best_idx] + [merged] + tokens[best_idx + 2:]
            tokens = new_tokens
        
        return tokens
    
    def encode(self, text: str) -> list[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
        
        Returns:
            List of token IDs
        """
        if not text:
            return []
        
        token_ids = []
        
        # Split by special tokens if we have any
        if self.special_pattern:
            chunks = self.special_pattern.split(text)
        else:
            chunks = [text]
        
        for chunk in chunks:
            if not chunk:
                continue
            
            # Check if this chunk is a special token
            if chunk in self.special_tokens:
                token_bytes = chunk.encode('utf-8')
                if token_bytes in self.token_to_id:
                    token_ids.append(self.token_to_id[token_bytes])
                continue
            
            # Pre-tokenize into words/pieces
            for match in self.pattern.finditer(chunk):
                piece = match.group()
                
                # Convert to bytes
                piece_bytes = piece.encode('utf-8')
                
                # Split into individual bytes
                tokens = [bytes([b]) for b in piece_bytes]
                
                # Apply BPE merges
                tokens = self._apply_merges(tokens)
                
                # Convert to IDs
                for token in tokens:
                    if token in self.token_to_id:
                        token_ids.append(self.token_to_id[token])
        
        return token_ids
    
    def decode(self, token_ids: list[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
        
        Returns:
            Decoded text string
        """
        # Convert IDs to bytes
        token_bytes = b''.join(self.vocab[idx] for idx in token_ids)
        
        # Decode to string
        text = token_bytes.decode('utf-8', errors='replace')
        
        return text
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        Encode an iterable of strings efficiently.
        
        This method is memory-efficient for large inputs.
        
        Args:
            iterable: Iterable of strings (e.g., file object)
        
        Yields:
            Token IDs one at a time
        """
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on a corpus.
    
    Args:
        input_path: Path to training corpus
        vocab_size: Desired vocabulary size
        special_tokens: Special tokens to add to vocabulary
    
    Returns:
        Tuple of (vocab, merges) where:
        - vocab: Mapping from token ID to token bytes
        - merges: List of BPE merges in order
    """
    # Pre-tokenization pattern (GPT-2 style)
    pattern = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
        re.IGNORECASE
    )
    
    # Read and pre-tokenize corpus
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Pre-tokenize into words
    words = pattern.findall(text)
    
    # Convert words to byte sequences
    word_counts = Counter()
    for word in words:
        word_bytes = word.encode('utf-8')
        word_counts[word_bytes] += 1
    
    # Initialize vocabulary with all bytes
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    
    # Add special tokens to vocab
    for special_token in special_tokens:
        special_bytes = special_token.encode('utf-8')
        vocab[next_id] = special_bytes
        next_id += 1
    
    # Track merges
    merges = []
    
    # Number of merges needed
    num_merges = vocab_size - len(vocab)
    
    # Split words into tokens (initially individual bytes)
    word_tokens = {}
    for word_bytes, count in word_counts.items():
        tokens = tuple(bytes([b]) for b in word_bytes)
        word_tokens[word_bytes] = (tokens, count)
    
    # Perform BPE merges
    for _ in range(num_merges):
        # Count all adjacent pairs
        pair_counts = Counter()
        for word_bytes, (tokens, count) in word_tokens.items():
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] += count
        
        if not pair_counts:
            break
        
        # Find most frequent pair
        best_pair = max(pair_counts, key=pair_counts.get)
        
        # Add merge
        merges.append(best_pair)
        
        # Add merged token to vocab
        merged_token = best_pair[0] + best_pair[1]
        vocab[next_id] = merged_token
        next_id += 1
        
        # Update word tokens with the merge
        new_word_tokens = {}
        for word_bytes, (tokens, count) in word_tokens.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                # Check if we can merge at this position
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_word_tokens[word_bytes] = (tuple(new_tokens), count)
        word_tokens = new_word_tokens
    
    return vocab, merges

