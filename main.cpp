/*
 * Highly optimized UCI Chess Engine & Parallel Perft Counter (improved)
 *
 * Main runtime optimizations:
 * - Avoid std::vector in the recursive perft hot-path: use fixed-size arrays.
 * - Undo-based make/unmake rather than copying the whole Board each move.
 * - Inline hot functions and reduce branching where straightforward.
 * - Reserve work per-thread at root; threads only copy the board once per root move.
 *
 * Compile: g++ -O3 -std=c++17 -march=native -pthread chess_perft_opt.cpp -o chess_perft_opt
 */

#include <iostream>
#include <string>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <array>

typedef uint64_t U64;

enum {
    a1, b1, c1, d1, e1, f1, g1, h1,
    a2, b2, c2, d2, e2, f2, g2, h2,
    a3, b3, c3, d3, e3, f3, g3, h3,
    a4, b4, c4, d4, e4, f4, g4, h4,
    a5, b5, c5, d5, e5, f5, g5, h5,
    a6, b6, c6, d6, e6, f6, g6, h6,
    a7, b7, c7, d7, e7, f7, g7, h7,
    a8, b8, c8, d8, e8, f8, g8, h8, no_sq
};

enum { white, black, both };
enum { P, N, B, R, Q, K, p, n, b, r, q, k };
enum { wk = 1, wq = 2, bk = 4, bq = 8 };

const char* square_to_coord[] = { "a1","b1","c1","d1","e1","f1","g1","h1",
    "a2","b2","c2","d2","e2","f2","g2","h2","a3","b3","c3","d3","e3","f3","g3","h3",
    "a4","b4","c4","d4","e4","f4","g4","h4","a5","b5","c5","d5","e5","f5","g5","h5",
    "a6","b6","c6","d6","e6","f6","g6","h6","a7","b7","c7","d7","e7","f7","g7","h7",
    "a8","b8","c8","d8","e8","f8","g8","h8"
};

// Bit helpers
#define get_bit(bb, sq) ((bb) & (1ULL << (sq)))
#define set_bit(bb, sq) ((bb) |= (1ULL << (sq)))
#define pop_bit(bb, sq) ((bb) &= ~(1ULL << (sq)))

static inline int pop_lsb(U64& bb) {
    int sq = __builtin_ctzll(bb);
    bb &= bb - 1;
    return sq;
}
static inline int count_bits(U64 bb) { return __builtin_popcountll(bb); }

// Attack tables (unchanged)
U64 pawn_attacks[2][64];
U64 knight_attacks[64];
U64 king_attacks[64];

U64 bishop_masks[64];
U64 rook_masks[64];
U64 bishop_attacks[64][512];
U64 rook_attacks[64][4096];

const U64 bishop_magics[64] = {
    0x40040844404084ULL, 0x2004208a004208ULL, 0x10190041080202ULL, 0x108060845042010ULL,
    0x581104180800210ULL, 0x2112080446200ULL, 0x1080820820060210ULL, 0x3c0808410220200ULL,
    0x4050404440404ULL, 0x21001420088ULL, 0x24d0080801082102ULL, 0x1020a0a020400ULL,
    0x40308200402ULL, 0x4011002100800ULL, 0x401484104104005ULL, 0x801010402020200ULL,
    0x400210c3880100ULL, 0x404022024108200ULL, 0x810018200204102ULL, 0x4002801a02003ULL,
    0x85040820080400ULL, 0x810102c808880400ULL, 0xe900410884800ULL, 0x8002020480840102ULL,
    0x220200865090201ULL, 0x2010100a02021202ULL, 0x152048408022401ULL, 0x20080002081110ULL,
    0x4001001021004000ULL, 0x800040400a011002ULL, 0xe4004081011002ULL, 0x1c004001012080ULL,
    0x8004200962a00220ULL, 0x8422100208500202ULL, 0x2000402200300c08ULL, 0x8646020080080080ULL,
    0x80020a0200100808ULL, 0x2010004880111000ULL, 0x623000a080011400ULL, 0x42008c0340209202ULL,
    0x209188240001000ULL, 0x400408a884001800ULL, 0x110400a6080400ULL, 0x1840060a44020800ULL,
    0x90080104000041ULL, 0x201011000808101ULL, 0x1a2208080504f080ULL, 0x8012020600211212ULL,
    0x500861011240000ULL, 0x180806108200800ULL, 0x4000020e01040044ULL, 0x300000261044000aULL,
    0x802241102020002ULL, 0x20906061210001ULL, 0x5a84841004010310ULL, 0x4010801011c04ULL,
    0xa010109502200ULL, 0x4a02012000ULL, 0x500201010098b028ULL, 0x8040002811040900ULL,
    0x28000010020204ULL, 0x6000020202d0240ULL, 0x8918844842082200ULL, 0x4010011029020020ULL
};

const U64 rook_magics[64] = {
    0x8a80104000800020ULL, 0x140002000100040ULL, 0x2801880a0017001ULL, 0x100081001000420ULL,
    0x200020010080420ULL, 0x3001c0002010008ULL, 0x8480008002000100ULL, 0x2080088004402900ULL,
    0x800098204000ULL, 0x2024401000200040ULL, 0x100802000801000ULL, 0x120800800801000ULL,
    0x208808088000400ULL, 0x2802200800400ULL, 0x2200800100020080ULL, 0x801000060821100ULL,
    0x80044006422000ULL, 0x100808020004000ULL, 0x12108a0010204200ULL, 0x140848010000802ULL,
    0x481828014002800ULL, 0x8094004002004100ULL, 0x4010040010010802ULL, 0x20008806104ULL,
    0x100400080208000ULL, 0x2040002120081000ULL, 0x21200680100081ULL, 0x20100080080080ULL,
    0x2000a00200410ULL, 0x20080800400ULL, 0x80088400100102ULL, 0x80004600042881ULL,
    0x4040008040800020ULL, 0x440003000200801ULL, 0x4200011004500ULL, 0x188020010100100ULL,
    0x14800401802800ULL, 0x2080040080800200ULL, 0x124080204001001ULL, 0x200046502000484ULL,
    0x480400080088020ULL, 0x1000422010034000ULL, 0x30200100110040ULL, 0x100021010009ULL,
    0x2002080100110004ULL, 0x202008004008002ULL, 0x20020004010100ULL, 0x2048440040820001ULL,
    0x101002200408200ULL, 0x40802000401080ULL, 0x4008142004410100ULL, 0x2060820c0120200ULL,
    0x1001004080100ULL, 0x20c020080040080ULL, 0x2935610830022400ULL, 0x44440041009200ULL,
    0x280001040802101ULL, 0x2100190040002085ULL, 0x80c0084100102001ULL, 0x4024081001000421ULL,
    0x20030a0244872ULL, 0x12001008414402ULL, 0x2006104900a0804ULL, 0x1004081002402ULL
};

// masks and on-fly attack generation (unchanged)
U64 mask_bishop_attacks(int sq) {
    U64 attacks = 0ULL;
    int r = sq / 8, f = sq % 8;
    for (int rr = r + 1, ff = f + 1; rr <= 6 && ff <= 6; rr++, ff++) set_bit(attacks, rr * 8 + ff);
    for (int rr = r + 1, ff = f - 1; rr <= 6 && ff >= 1; rr++, ff--) set_bit(attacks, rr * 8 + ff);
    for (int rr = r - 1, ff = f + 1; rr >= 1 && ff <= 6; rr--, ff++) set_bit(attacks, rr * 8 + ff);
    for (int rr = r - 1, ff = f - 1; rr >= 1 && ff >= 1; rr--, ff--) set_bit(attacks, rr * 8 + ff);
    return attacks;
}

U64 mask_rook_attacks(int sq) {
    U64 attacks = 0ULL;
    int r = sq / 8, f = sq % 8;
    for (int rr = r + 1; rr <= 6; rr++) set_bit(attacks, rr * 8 + f);
    for (int rr = r - 1; rr >= 1; rr--) set_bit(attacks, rr * 8 + f);
    for (int ff = f + 1; ff <= 6; ff++) set_bit(attacks, r * 8 + ff);
    for (int ff = f - 1; ff >= 1; ff--) set_bit(attacks, r * 8 + ff);
    return attacks;
}

U64 bishop_attacks_on_fly(int sq, U64 block) {
    U64 attacks = 0ULL;
    int r = sq / 8, f = sq % 8;
    for (int rr = r + 1, ff = f + 1; rr <= 7 && ff <= 7; rr++, ff++) {
        set_bit(attacks, rr * 8 + ff);
        if (get_bit(block, rr * 8 + ff)) break;
    }
    for (int rr = r + 1, ff = f - 1; rr <= 7 && ff >= 0; rr++, ff--) {
        set_bit(attacks, rr * 8 + ff);
        if (get_bit(block, rr * 8 + ff)) break;
    }
    for (int rr = r - 1, ff = f + 1; rr >= 0 && ff <= 7; rr--, ff++) {
        set_bit(attacks, rr * 8 + ff);
        if (get_bit(block, rr * 8 + ff)) break;
    }
    for (int rr = r - 1, ff = f - 1; rr >= 0 && ff >= 0; rr--, ff--) {
        set_bit(attacks, rr * 8 + ff);
        if (get_bit(block, rr * 8 + ff)) break;
    }
    return attacks;
}

U64 rook_attacks_on_fly(int sq, U64 block) {
    U64 attacks = 0ULL;
    int r = sq / 8, f = sq % 8;
    for (int rr = r + 1; rr <= 7; rr++) {
        set_bit(attacks, rr * 8 + f);
        if (get_bit(block, rr * 8 + f)) break;
    }
    for (int rr = r - 1; rr >= 0; rr--) {
        set_bit(attacks, rr * 8 + f);
        if (get_bit(block, rr * 8 + f)) break;
    }
    for (int ff = f + 1; ff <= 7; ff++) {
        set_bit(attacks, r * 8 + ff);
        if (get_bit(block, r * 8 + ff)) break;
    }
    for (int ff = f - 1; ff >= 0; ff--) {
        set_bit(attacks, r * 8 + ff);
        if (get_bit(block, r * 8 + ff)) break;
    }
    return attacks;
}

void init_sliders() {
    for (int sq = 0; sq < 64; sq++) {
        bishop_masks[sq] = mask_bishop_attacks(sq);
        rook_masks[sq] = mask_rook_attacks(sq);

        int bishop_bits = count_bits(bishop_masks[sq]);
        int rook_bits = count_bits(rook_masks[sq]);

        // bishop
        int bishop_variations = 1 << bishop_bits;
        for (int i = 0; i < bishop_variations; ++i) {
            U64 occ = 0ULL;
            U64 mask = bishop_masks[sq];
            int count = 0;
            U64 tmp = mask;
            while (tmp) {
                int bit = pop_lsb(tmp);
                if (i & (1 << count)) set_bit(occ, bit);
                ++count;
            }
            int magic_index = (int)((occ * bishop_magics[sq]) >> (64 - 9));
            bishop_attacks[sq][magic_index] = bishop_attacks_on_fly(sq, occ);
        }

        // rook
        int rook_variations = 1 << rook_bits;
        for (int i = 0; i < rook_variations; ++i) {
            U64 occ = 0ULL;
            U64 mask = rook_masks[sq];
            int count = 0;
            U64 tmp = mask;
            while (tmp) {
                int bit = pop_lsb(tmp);
                if (i & (1 << count)) set_bit(occ, bit);
                ++count;
            }
            int magic_index = (int)((occ * rook_magics[sq]) >> (64 - 12));
            rook_attacks[sq][magic_index] = rook_attacks_on_fly(sq, occ);
        }
    }
}

inline U64 get_bishop_attacks(int sq, U64 occ) {
    occ &= bishop_masks[sq];
    occ *= bishop_magics[sq];
    occ >>= 64 - 9;
    return bishop_attacks[sq][occ];
}
inline U64 get_rook_attacks(int sq, U64 occ) {
    occ &= rook_masks[sq];
    occ *= rook_magics[sq];
    occ >>= 64 - 12;
    return rook_attacks[sq][occ];
}
inline U64 get_queen_attacks(int sq, U64 occ) {
    return get_bishop_attacks(sq, occ) | get_rook_attacks(sq, occ);
}

void init_leapers() {
    for (int sq = 0; sq < 64; ++sq) {
        pawn_attacks[white][sq] = pawn_attacks[black][sq] = 0ULL;
        knight_attacks[sq] = king_attacks[sq] = 0ULL;
    }

    for (int sq = 0; sq < 64; ++sq) {
        int r = sq / 8, f = sq % 8;
        if (r < 7) {
            if (f > 0) set_bit(pawn_attacks[white][sq], sq + 7);
            if (f < 7) set_bit(pawn_attacks[white][sq], sq + 9);
        }
        if (r > 0) {
            if (f > 0) set_bit(pawn_attacks[black][sq], sq - 9);
            if (f < 7) set_bit(pawn_attacks[black][sq], sq - 7);
        }

        const int n_offsets[] = {17,15,10,6,-6,-10,-15,-17};
        for (int off : n_offsets) {
            int target = sq + off;
            if (target >= 0 && target < 64) {
                int dr = std::abs(r - target/8), df = std::abs(f - target%8);
                if (dr + df == 3 && dr != 0 && df != 0) set_bit(knight_attacks[sq], target);
            }
        }

        const int k_offsets[] = {8,-8,1,-1,9,7,-7,-9};
        for (int off : k_offsets) {
            int target = sq + off;
            if (target >= 0 && target < 64 && std::abs(r - target/8) <= 1 && std::abs(f - target%8) <= 1)
                set_bit(king_attacks[sq], target);
        }
    }
}

// Board & Move
struct Move {
    int source;
    int target;
    int promoted;      // -1 if none, else piece index (P..K or p..k)
    bool capture;
    bool double_push;
    bool en_passant;
    bool castling;
};

static inline std::string move_to_string(const Move& m) {
    std::string s = std::string(square_to_coord[m.source]) + square_to_coord[m.target];
    if (m.promoted != -1) {
        int p = (m.promoted > 6 ? m.promoted - 6 : m.promoted);
        s += (p == N ? 'n' : p == B ? 'b' : p == R ? 'r' : 'q');
    }
    return s;
}

class Board {
public:
    U64 bitboards[12];
    U64 occupancies[3];
    int side;
    int en_passant;
    int castle;

    Board() { reset(); }

    inline void reset() {
        memset(bitboards, 0, sizeof(bitboards));
        memset(occupancies, 0, sizeof(occupancies));
        side = white;
        en_passant = no_sq;
        castle = 0;
    }

    inline void update_occupancies() {
        occupancies[white] = bitboards[P] | bitboards[N] | bitboards[B] | bitboards[R] | bitboards[Q] | bitboards[K];
        occupancies[black] = bitboards[p] | bitboards[n] | bitboards[b] | bitboards[r] | bitboards[q] | bitboards[k];
        occupancies[both]  = occupancies[white] | occupancies[black];
    }

    inline bool is_square_attacked(int sq, int by_side) const {
        if (by_side == white) {
            if (pawn_attacks[black][sq] & bitboards[P]) return true;
            if (knight_attacks[sq] & bitboards[N]) return true;
            if (king_attacks[sq] & bitboards[K]) return true;
            U64 bishops_queens = bitboards[B] | bitboards[Q];
            if (bishops_queens && (get_bishop_attacks(sq, occupancies[both]) & bishops_queens)) return true;
            U64 rooks_queens = bitboards[R] | bitboards[Q];
            if (rooks_queens && (get_rook_attacks(sq, occupancies[both]) & rooks_queens)) return true;
        } else {
            if (pawn_attacks[white][sq] & bitboards[p]) return true;
            if (knight_attacks[sq] & bitboards[n]) return true;
            if (king_attacks[sq] & bitboards[k]) return true;
            U64 bishops_queens = bitboards[b] | bitboards[q];
            if (bishops_queens && (get_bishop_attacks(sq, occupancies[both]) & bishops_queens)) return true;
            U64 rooks_queens = bitboards[r] | bitboards[q];
            if (rooks_queens && (get_rook_attacks(sq, occupancies[both]) & rooks_queens)) return true;
        }
        return false;
    }

    void parse_fen(const std::string& fen) {
        reset();
        std::stringstream ss(fen);
        std::string placement, turn, castling, ep_str;
        ss >> placement >> turn >> castling >> ep_str;

        int rank = 7, file = 0;
        for (char c : placement) {
            if (c == '/') { rank--; file = 0; }
            else if (isdigit(c)) { file += (c - '0'); }
            else {
                int sq = rank * 8 + file;
                int piece = -1;
                switch(c) {
                    case 'P': piece = P; break; case 'N': piece = N; break;
                    case 'B': piece = B; break; case 'R': piece = R; break;
                    case 'Q': piece = Q; break; case 'K': piece = K; break;
                    case 'p': piece = p; break; case 'n': piece = n; break;
                    case 'b': piece = b; break; case 'r': piece = r; break;
                    case 'q': piece = q; break; case 'k': piece = k; break;
                }
                if (piece != -1) set_bit(bitboards[piece], sq);
                ++file;
            }
        }
        side = (turn == "w") ? white : black;
        castle = 0;
        if (castling != "-") {
            if (castling.find('K') != std::string::npos) castle |= wk;
            if (castling.find('Q') != std::string::npos) castle |= wq;
            if (castling.find('k') != std::string::npos) castle |= bk;
            if (castling.find('q') != std::string::npos) castle |= bq;
        }
        if (ep_str != "-") en_passant = (ep_str[1] - '1') * 8 + (ep_str[0] - 'a');
        else en_passant = no_sq;
        update_occupancies();
    }
};

// Undo record (small)
struct Undo {
    int move_piece;
    int captured_piece;
    int captured_sq;
    int prev_en_passant;
    int prev_castle;
    bool was_promotion;
    bool was_castling;
};

// Move generation: fill a fixed array, returns count. Avoid dynamic alloc.
inline void add_pawn_moves_static(Move* moves, int& idx, int src, int tgt, bool is_capture, bool is_promo, int promo_start, bool gen_caps, bool gen_quiets) {
    if (is_promo) {
        if (gen_caps && is_capture) {
            moves[idx++] = {src,tgt,promo_start+4,true,false,false,false};
            moves[idx++] = {src,tgt,promo_start+3,true,false,false,false};
            moves[idx++] = {src,tgt,promo_start+2,true,false,false,false};
            moves[idx++] = {src,tgt,promo_start+1,true,false,false,false};
        } else if (gen_quiets && !is_capture) {
            moves[idx++] = {src,tgt,promo_start+4,false,false,false,false};
            moves[idx++] = {src,tgt,promo_start+3,false,false,false,false};
            moves[idx++] = {src,tgt,promo_start+2,false,false,false,false};
            moves[idx++] = {src,tgt,promo_start+1,false,false,false,false};
        }
    } else {
        if ((gen_caps && is_capture) || (gen_quiets && !is_capture)) {
            moves[idx++] = {src,tgt,-1,is_capture,false,false,false};
        }
    }
}

// generate_moves writes to moves[] and returns count (no allocations)
int generate_moves(const Board& board, Move* moves) {
    int mcount = 0;
    int us = board.side;
    int them = us ^ 1;
    U64 our_pieces = board.occupancies[us];
    U64 their_pieces = board.occupancies[them];
    U64 empty = ~board.occupancies[both];

    if (us == white) {
        U64 pawns = board.bitboards[P];
        while (pawns) {
            int src = pop_lsb(pawns);
            int tgt = src + 8;
            bool promo = src >= a7;
            if (tgt < 64 && get_bit(empty, tgt)) {
                add_pawn_moves_static(moves, mcount, src, tgt, false, promo, P, false, true);
                if (src >= a2 && src <= h2 && get_bit(empty, src + 16)) {
                    moves[mcount++] = {src, src + 16, -1, false, true, false, false};
                }
            }
            U64 attacks = pawn_attacks[white][src] & their_pieces;
            while (attacks) {
                int t = pop_lsb(attacks);
                add_pawn_moves_static(moves, mcount, src, t, true, promo, P, true, false);
            }
            if (board.en_passant != no_sq && (pawn_attacks[white][src] & (1ULL << board.en_passant))) {
                moves[mcount++] = {src, board.en_passant, -1, true, false, true, false};
            }
        }
        // castling
        if ((board.castle & wk) && !(board.occupancies[both] & 0x60ULL)) {
            if (!board.is_square_attacked(e1, black) && !board.is_square_attacked(f1, black))
                moves[mcount++] = {e1, g1, -1, false, false, false, true};
        }
        if ((board.castle & wq) && !(board.occupancies[both] & 0xEULL)) {
            if (!board.is_square_attacked(e1, black) && !board.is_square_attacked(d1, black))
                moves[mcount++] = {e1, c1, -1, false, false, false, true};
        }
    } else {
        U64 pawns = board.bitboards[p];
        while (pawns) {
            int src = pop_lsb(pawns);
            int tgt = src - 8;
            bool promo = src >= a2 && src <= h2;
            if (tgt >= 0 && get_bit(empty, tgt)) {
                add_pawn_moves_static(moves, mcount, src, tgt, false, promo, p, false, true);
                if (src >= a7 && src <= h7 && get_bit(empty, src - 16)) {
                    moves[mcount++] = {src, src - 16, -1, false, true, false, false};
                }
            }
            U64 attacks = pawn_attacks[black][src] & their_pieces;
            while (attacks) {
                int t = pop_lsb(attacks);
                add_pawn_moves_static(moves, mcount, src, t, true, promo, p, true, false);
            }
            if (board.en_passant != no_sq && (pawn_attacks[black][src] & (1ULL << board.en_passant))) {
                moves[mcount++] = {src, board.en_passant, -1, true, false, true, false};
            }
        }
        if ((board.castle & bk) && !(board.occupancies[both] & 0x6000000000000000ULL)) {
            if (!board.is_square_attacked(e8, white) && !board.is_square_attacked(f8, white))
                moves[mcount++] = {e8, g8, -1, false, false, false, true};
        }
        if ((board.castle & bq) && !(board.occupancies[both] & 0xE00000000000000ULL)) {
            if (!board.is_square_attacked(e8, white) && !board.is_square_attacked(d8, white))
                moves[mcount++] = {e8, c8, -1, false, false, false, true};
        }
    }

    // Non-pawn pieces
    const int pieces[5] = { (us==white?N:n), (us==white?B:b), (us==white?R:r), (us==white?Q:q), (us==white?K:k) };
    for (int pt = 0; pt < 5; ++pt) {
        int piece = pieces[pt];
        U64 bb = board.bitboards[piece];
        while (bb) {
            int src = pop_lsb(bb);
            U64 attacks = 0ULL;
            if (piece == N || piece == n) attacks = knight_attacks[src];
            else if (piece == B || piece == b) attacks = get_bishop_attacks(src, board.occupancies[both]);
            else if (piece == R || piece == r) attacks = get_rook_attacks(src, board.occupancies[both]);
            else if (piece == Q || piece == q) attacks = get_queen_attacks(src, board.occupancies[both]);
            else attacks = king_attacks[src];

            attacks &= ~our_pieces;
            while (attacks) {
                int tgt = pop_lsb(attacks);
                bool cap = (get_bit(their_pieces, tgt) != 0);
                moves[mcount++] = {src, tgt, -1, cap, false, false, false};
            }
        }
    }
    return mcount;
}

// Fast helper: locate piece index at a square for side 'side'
inline int piece_on_square(const Board& board, int side, int sq) {
    int start = (side == white) ? P : p;
    int end   = (side == white) ? K : k;
    for (int i = start; i <= end; ++i) if (get_bit(board.bitboards[i], sq)) return i;
    return -1;
}

// make_move with undo (no full board copy). Returns true if legal, false if move leaves king in check.
inline bool make_move(Board& board, const Move& move, Undo& undo) {
    // save undo basics
    undo.prev_en_passant = board.en_passant;
    undo.prev_castle = board.castle;
    undo.was_promotion = (move.promoted != -1);
    undo.was_castling = move.castling;
    undo.captured_piece = -1;
    undo.captured_sq = -1;

    int side = board.side;
    int piece_start = (side == white) ? P : p;
    int piece_end   = (side == white) ? K : k;

    // find moving piece index
    int move_piece = -1;
    for (int i = piece_start; i <= piece_end; ++i) {
        if (get_bit(board.bitboards[i], move.source)) { move_piece = i; break; }
    }
    if (move_piece == -1) return false; // invalid

    undo.move_piece = move_piece;

    // Remove from source
    pop_bit(board.bitboards[move_piece], move.source);

    // Handle captures
    if (move.capture) {
        if (move.en_passant) {
            int victim_sq = (side == white) ? move.target - 8 : move.target + 8;
            int victim_piece = (side == white) ? p : P;
            // victim is pawn by definition
            pop_bit(board.bitboards[victim_piece], victim_sq);
            undo.captured_piece = victim_piece;
            undo.captured_sq = victim_sq;
        } else {
            // find captured piece at target
            int cap_start = (side == white) ? p : P;
            int cap_end   = (side == white) ? k : K;
            for (int i = cap_start; i <= cap_end; ++i) {
                if (get_bit(board.bitboards[i], move.target)) {
                    pop_bit(board.bitboards[i], move.target);
                    undo.captured_piece = i;
                    undo.captured_sq = move.target;
                    break;
                }
            }
        }
    }

    // Place moving piece (or promotion)
    if (move.promoted != -1) {
        // place promoted piece
        set_bit(board.bitboards[move.promoted], move.target);
    } else {
        set_bit(board.bitboards[move_piece], move.target);
    }

    // Castling rook move
    if (move.castling) {
        if (move.target == g1) { pop_bit(board.bitboards[R], h1); set_bit(board.bitboards[R], f1); }
        else if (move.target == c1) { pop_bit(board.bitboards[R], a1); set_bit(board.bitboards[R], d1); }
        else if (move.target == g8) { pop_bit(board.bitboards[r], h8); set_bit(board.bitboards[r], f8); }
        else if (move.target == c8) { pop_bit(board.bitboards[r], a8); set_bit(board.bitboards[r], d8); }
    }

    // Update castling rights
    if (move_piece == K || move_piece == k) {
        board.castle &= (side == white) ? ~(wk | wq) : ~(bk | bq);
    } else if (move_piece == R || move_piece == r) {
        // if rook moved from corner, remove right
        if (move.source == h1) board.castle &= ~wk;
        else if (move.source == a1) board.castle &= ~wq;
        else if (move.source == h8) board.castle &= ~bk;
        else if (move.source == a8) board.castle &= ~bq;
    }
    // if capture target was rook from original corner
    if (undo.captured_piece == R) {
        if (undo.captured_sq == h1) board.castle &= ~wk;
        else if (undo.captured_sq == a1) board.castle &= ~wq;
    } else if (undo.captured_piece == r) {
        if (undo.captured_sq == h8) board.castle &= ~bk;
        else if (undo.captured_sq == a8) board.castle &= ~bq;
    }

    // Set en_passant square
    board.en_passant = move.double_push ? (side == white ? move.target - 8 : move.target + 8) : no_sq;

    board.update_occupancies();

    // King safety
    int k_sq = __builtin_ctzll(board.bitboards[(side==white)?K:k]);
    if (board.is_square_attacked(k_sq, side ^ 1)) {
        // undo everything quickly
        // remove piece from target
        if (move.promoted != -1) pop_bit(board.bitboards[move.promoted], move.target);
        else pop_bit(board.bitboards[move_piece], move.target);

        // restore moving piece to source
        set_bit(board.bitboards[move_piece], move.source);

        // restore captured piece if any
        if (undo.captured_piece != -1) set_bit(board.bitboards[undo.captured_piece], undo.captured_sq);

        // undo castling rook if applied
        if (move.castling) {
            if (move.target == g1) { set_bit(board.bitboards[R], h1); pop_bit(board.bitboards[R], f1); }
            else if (move.target == c1) { set_bit(board.bitboards[R], a1); pop_bit(board.bitboards[R], d1); }
            else if (move.target == g8) { set_bit(board.bitboards[r], h8); pop_bit(board.bitboards[r], f8); }
            else if (move.target == c8) { set_bit(board.bitboards[r], a8); pop_bit(board.bitboards[r], d8); }
        }

        // restore metadata
        board.en_passant = undo.prev_en_passant;
        board.castle = undo.prev_castle;
        board.update_occupancies();
        return false;
    }

    board.side = side ^ 1;
    return true;
}

inline void unmake_move(Board& board, const Move& move, const Undo& undo) {
    board.side ^= 1; // side was toggled in make_move

    // remove piece from target
    if (move.promoted != -1) pop_bit(board.bitboards[move.promoted], move.target);
    else {
        pop_bit(board.bitboards[undo.move_piece], move.target);
    }

    // restore move piece to source
    set_bit(board.bitboards[undo.move_piece], move.source);

    // restore captured piece if present
    if (undo.captured_piece != -1) {
        set_bit(board.bitboards[undo.captured_piece], undo.captured_sq);
    }

    // undo castling rook move
    if (undo.was_castling) {
        if (move.target == g1) { set_bit(board.bitboards[R], h1); pop_bit(board.bitboards[R], f1); }
        else if (move.target == c1) { set_bit(board.bitboards[R], a1); pop_bit(board.bitboards[R], d1); }
        else if (move.target == g8) { set_bit(board.bitboards[r], h8); pop_bit(board.bitboards[r], f8); }
        else if (move.target == c8) { set_bit(board.bitboards[r], a8); pop_bit(board.bitboards[r], d8); }
    }

    // restore en_passant and castle
    board.en_passant = undo.prev_en_passant;
    board.castle = undo.prev_castle;
    board.update_occupancies();
}

// perft recursive driver using static arrays
uint64_t perft_driver(Board& board, int depth) {
    if (depth == 0) return 1ULL;

    Move moves[256];
    int n = generate_moves(board, moves);
    uint64_t nodes = 0ULL;

    Undo undo;
    for (int i = 0; i < n; ++i) {
        if (make_move(board, moves[i], undo)) {
            nodes += perft_driver(board, depth - 1);
            unmake_move(board, moves[i], undo);
        }
    }
    return nodes;
}

void perft_test(Board& board, int depth) {
    std::cout << "Performance test (depth " << depth << ") [6 Threads]\n";
    Move root_moves[256];
    int num_moves = generate_moves(board, root_moves);

    std::atomic<int> current_move_idx(0);
    std::atomic<uint64_t> total_nodes(0);
    std::mutex print_mutex;

    auto start = std::chrono::high_resolution_clock::now();

    auto worker = [&](int) {
        while (true) {
            int idx = current_move_idx.fetch_add(1);
            if (idx >= num_moves) break;
            Move mv = root_moves[idx];
            Board local = board;
            Undo undo;
            if (!make_move(local, mv, undo)) continue;
            uint64_t nodes = perft_driver(local, depth - 1);
            {
                std::lock_guard<std::mutex> lock(print_mutex);
                std::cout << move_to_string(mv) << ": " << nodes << "\n";
            }
            total_nodes += nodes;
        }
    };

    const int THREADS = 6;
    std::vector<std::thread> threads;
    threads.reserve(THREADS);
    for (int t = 0; t < THREADS; ++t) threads.emplace_back(worker, t);
    for (auto& th : threads) th.join();

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "\nNodes: " << total_nodes.load() << "\n";
    std::cout << "Time: " << ms.count() << "ms\n";
}

int main() {
    init_leapers();
    init_sliders();

    Board board;
    board.parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    std::string line, token;
    while (std::getline(std::cin, line)) {
        std::stringstream ss(line);
        ss >> token;
        if (token == "uci") {
            std::cout << "id name OptimizedPerft\nid author Optimized\nuciok\n";
        } else if (token == "isready") std::cout << "readyok\n";
        else if (token == "position") {
            ss >> token;
            if (token == "startpos") {
                board.parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
                ss >> token;
            } else if (token == "fen") {
                std::string fen;
                while (ss >> token && token != "moves") fen += token + " ";
                board.parse_fen(fen);
            }
            while (ss >> token) {
                Move moves[256];
                int n = generate_moves(board, moves);
                for (int i = 0; i < n; ++i) {
                    if (move_to_string(moves[i]) == token) {
                        Undo undo;
                        make_move(board, moves[i], undo);
                        break;
                    }
                }
            }
        } else if (token == "go") {
            ss >> token;
            if (token == "perft") {
                int depth; ss >> depth;
                perft_test(board, depth);
            }
        } else if (token == "quit") break;
    }
    return 0;
}
