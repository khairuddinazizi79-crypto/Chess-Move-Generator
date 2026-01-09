#Chess Move Generator

A high-performance chess move generator and perft counter written in modern C++ as a single-file project.
The focus is on speed, clean bitboard-based design, and correctness.

This project is **not a full chess engine** (no evaluation or search), but it implements:

- Fully legal move generation
- Fast make/unmake logic
- Multi-threaded perft testing
- A minimal UCI-compatible interface for testing

## Features

- Bitboard-based board representation
- Magic bitboards for bishop and rook attacks
- Precomputed attack tables for:
  - Pawns
  - Knights
  - Kings
- Fully legal move generation:
  - Captures
  - Promotions
  - Double pawn pushes
  - En passant
  - Castling
- Legality checking (king-in-check filtering)
- Optimized `make_move`
  - Early exits
  - Minimal branching
- Parallel perft at root
  - Fixed 6-thread implementation
- Single source file, no dependencies


## Usage

The program reads commands from **stdin** and understands a subset of the **UCI protocol**, mainly for perft testing.

### Basic UCI commands
```text
uci
isready

Set a position

Start position:

position startpos

Custom FEN:

position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1

With moves:

position startpos moves e2e4 e7e5 g1f3

Output format

e2e4: ...
d2d4: ...
...
Nodes: 4865609
Time: 123ms


Technical Overview

Board Representation
	•	12 piece bitboards (P..K, p..k)
	•	Occupancy bitboards:
	•	White
	•	Black
	•	Both
	•	Side to move, castling rights, en passant square


Move Representation

Each move stores:
	•	Source square
	•	Target square
	•	Promotion piece (if any)
	•	Capture flag
	•	Double pawn push
	•	En passant
	•	Castling

Moves are converted to UCI notation (e.g. e2e4, a7a8q).


Move Generation
	•	Pawns handled separately for clarity and speed
	•	Sliding pieces use magic bitboards
	•	Knights and kings use precomputed attack tables
	•	Castling legality checks:
	•	Empty squares
	•	No king-in-check traversal

All generated moves are legal (king safety verified after make).


Perft Implementation
	•	Recursive depth-first perft
	•	Root moves distributed across 6 worker threads
	•	Thread-safe printing and node accumulation
	•	Each thread works on its own board copy


Design Goals
	•	Maximum performance with readable code
	•	Educational reference for:
	•	Bitboards
	•	Magic bitboards
	•	Move generation
	•	Perft validation
	•	Easy to extend into a full chess engine


Notes
	•	No evaluation or search implemented
	•	No transposition tables
	•	Thread count is currently fixed to 6
	•	Intended primarily for experimentation, learning, and benchmarking
