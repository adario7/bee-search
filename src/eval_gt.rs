use crate::eval::Eval;

// Include trained weights
include!("../logs/eval_gt.rs");

#[inline(always)]
fn softsign(x: f32) -> f32 {
    x / (1.0 + x.abs())
}

#[inline(always)]
fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

#[inline(always)]
fn layer_norm(x: &mut [f32; GT_D_MODEL], gamma: &[f32; GT_D_MODEL], beta: &[f32; GT_D_MODEL]) {
    let mut sum = 0.0f32;
    for &v in x.iter() {
        sum += v;
    }
    let mean = sum / (GT_D_MODEL as f32);

    let mut var_sum = 0.0f32;
    for &v in x.iter() {
        let diff = v - mean;
        var_sum += diff * diff;
    }
    let inv_std = 1.0 / (var_sum / (GT_D_MODEL as f32) + 1e-5).sqrt();

    for i in 0..GT_D_MODEL {
        x[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

#[inline(always)]
fn compute_token_h0_and_qkv(
    tok: usize,
    feat: &[f32; GT_FEAT_DIM],
    h0_tok: &mut [f32; GT_D_MODEL],
    q: &mut [[[f32; GT_D_K]; GT_NUM_TOKENS]; GT_N_HEADS],
    k: &mut [[[f32; GT_D_K]; GT_NUM_TOKENS]; GT_N_HEADS],
    v: &mut [[[f32; GT_D_K]; GT_NUM_TOKENS]; GT_N_HEADS],
) {
    let is_on_board = feat[0] > 0.5;
    if GT_ZERO_IN_HAND && !is_on_board {
        *h0_tok = [0.0; GT_D_MODEL];
        for head in 0..GT_N_HEADS {
            q[head][tok] = [0.0; GT_D_K];
            k[head][tok] = [0.0; GT_D_K];
            v[head][tok] = [0.0; GT_D_K];
        }
        return;
    }

    // 1. Initial Token Projection
    for d in 0..GT_D_MODEL {
        let mut sum = GT_B_PROJ[tok][d];
        for f in 0..GT_FEAT_DIM {
            sum += feat[f] * GT_W_PROJ[tok][f][d];
        }
        h0_tok[d] = sum;
    }

    // 2. Layer 0 Q, K, V Projections
    for head in 0..GT_N_HEADS {
        for dk in 0..GT_D_K {
            let col = head * GT_D_K + dk;
            let mut sum_q = 0.0f32;
            let mut sum_k = 0.0f32;
            let mut sum_v = 0.0f32;
            for d in 0..GT_D_MODEL {
                let val = h0_tok[d];
                sum_q += val * GT_L0_WQ[tok][d][col];
                sum_k += val * GT_L0_WK[tok][d][col];
                sum_v += val * GT_L0_WV[tok][d][col];
            }
            q[head][tok][dk] = sum_q;
            k[head][tok][dk] = sum_k;
            v[head][tok][dk] = sum_v;
        }
    }
}

/// TokenCache supports incremental re-evaluation during tree search.
/// The first layer token embeddings h0 and Layer-0 Q, K, V for unmoved/unchanged pieces are cached.
#[derive(Clone, Debug)]
pub struct TokenCache {
    pub prev_features: [[f32; GT_FEAT_DIM]; GT_NUM_TOKENS],
    pub h0: [[f32; GT_D_MODEL]; GT_NUM_TOKENS],
    pub l0_q: [[[f32; GT_D_K]; GT_NUM_TOKENS]; GT_N_HEADS],
    pub l0_k: [[[f32; GT_D_K]; GT_NUM_TOKENS]; GT_N_HEADS],
    pub l0_v: [[[f32; GT_D_K]; GT_NUM_TOKENS]; GT_N_HEADS],
    pub placed: [usize; GT_NUM_TOKENS],
    pub n_placed: usize,
    pub initialized: bool,
}

impl TokenCache {
    pub fn new() -> Self {
        Self {
            prev_features: [[f32::NAN; GT_FEAT_DIM]; GT_NUM_TOKENS],
            h0: [[0.0; GT_D_MODEL]; GT_NUM_TOKENS],
            l0_q: [[[0.0; GT_D_K]; GT_NUM_TOKENS]; GT_N_HEADS],
            l0_k: [[[0.0; GT_D_K]; GT_NUM_TOKENS]; GT_N_HEADS],
            l0_v: [[[0.0; GT_D_K]; GT_NUM_TOKENS]; GT_N_HEADS],
            placed: [0; GT_NUM_TOKENS],
            n_placed: 0,
            initialized: false,
        }
    }

    pub fn reset(&mut self) {
        self.prev_features = [[f32::NAN; GT_FEAT_DIM]; GT_NUM_TOKENS];
        self.n_placed = 0;
        self.initialized = false;
    }

    pub fn update(&mut self, features: &[[f32; GT_FEAT_DIM]; GT_NUM_TOKENS]) {
        let mut n_placed = 0;
        for i in 0..GT_NUM_TOKENS {
            let is_on_board = features[i][0] > 0.5;
            if !GT_ZERO_IN_HAND || is_on_board {
                self.placed[n_placed] = i;
                n_placed += 1;
            }

            if !self.initialized || features[i] != self.prev_features[i] {
                compute_token_h0_and_qkv(
                    i,
                    &features[i],
                    &mut self.h0[i],
                    &mut self.l0_q,
                    &mut self.l0_k,
                    &mut self.l0_v,
                );
                self.prev_features[i] = features[i];
            }
        }
        self.n_placed = n_placed;
        self.initialized = true;
    }

    pub fn update_and_infer(
        &mut self,
        features: &[[f32; GT_FEAT_DIM]; GT_NUM_TOKENS],
        adj: &[[u8; GT_NUM_TOKENS]; GT_NUM_TOKENS],
    ) -> Eval {
        self.update(features);
        gt_inference_with_cache(self, adj)
    }
}

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use crate::graph_nn::TokenGraph;

static ACCUMULATE_ENABLED: AtomicBool = AtomicBool::new(false);
static ACCUMULATED_GRAPHS: Mutex<Vec<TokenGraph>> = Mutex::new(Vec::new());

pub fn set_accumulation(enabled: bool) {
    ACCUMULATE_ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn take_accumulated_graphs() -> Vec<TokenGraph> {
    let mut lock = ACCUMULATED_GRAPHS.lock().unwrap();
    std::mem::take(&mut *lock)
}

thread_local! {
    static TLS_TOKEN_CACHE: std::cell::RefCell<TokenCache> = std::cell::RefCell::new(TokenCache::new());
}

/// Reset thread-local token cache (e.g. on new game or new search)
pub fn reset_token_cache() {
    TLS_TOKEN_CACHE.with(|cache| {
        cache.borrow_mut().reset();
    });
}

/// Full CPU Graph Transformer static evaluation with zero heap allocation
pub fn gt_inference(
    features: &[[f32; GT_FEAT_DIM]; GT_NUM_TOKENS],
    adj: &[[u8; GT_NUM_TOKENS]; GT_NUM_TOKENS],
) -> Eval {
    let mut cache = TokenCache::new();
    cache.update_and_infer(features, adj)
}

/// Fast thread-local cached inference: only dirty / changed tokens recompute projections and Layer-0 QKV
pub fn gt_inference_fast(
    features: &[[f32; GT_FEAT_DIM]; GT_NUM_TOKENS],
    adj: &[[u8; GT_NUM_TOKENS]; GT_NUM_TOKENS],
) -> Eval {
    if ACCUMULATE_ENABLED.load(Ordering::Relaxed) {
        let tg = TokenGraph {
            features: *features,
            adj: *adj,
        };
        ACCUMULATED_GRAPHS.lock().unwrap().push(tg);
    }
    TLS_TOKEN_CACHE.with(|cache| {
        cache.borrow_mut().update_and_infer(features, adj)
    })
}

/// Computes evaluation given token cache and adjacency matrix
pub fn gt_inference_with_cache(
    cache: &TokenCache,
    adj: &[[u8; GT_NUM_TOKENS]; GT_NUM_TOKENS],
) -> Eval {
    let mut h = cache.h0;
    let placed = &cache.placed[..cache.n_placed];

    // Macro to execute attention + residual + LayerNorm1 + FFN + LayerNorm2
    macro_rules! run_attention_and_ffn {
        ($layer_h:ident, $q:expr, $k:expr, $v:expr, $wo:ident, $bo:ident, $eb:ident, $ng1:ident, $nb1:ident, $fw1:ident, $fb1:ident, $fw2:ident, $fb2:ident, $ng2:ident, $nb2:ident) => {
            let scale = 1.0 / (GT_D_K as f32).sqrt();
            let mut attn_out = [[0.0f32; GT_D_MODEL]; GT_NUM_TOKENS];

            for head in 0..GT_N_HEADS {
                for &i in placed {
                    let mut scores = [0.0f32; GT_NUM_TOKENS];
                    let mut denom = 1e-6f32;

                    for &j in placed {
                        let edge_cat = adj[i][j] as usize;
                        if GT_SPARSE_ATTN && edge_cat == 0 {
                            continue;
                        }

                        let mut dot = 0.0f32;
                        for dk in 0..GT_D_K {
                            dot += $q[head][i][dk] * $k[head][j][dk];
                        }
                        let s = dot * scale + $eb[edge_cat][head];
                        let r = relu(s);
                        let s_sq = r * r;
                        scores[j] = s_sq;
                        denom += s_sq;
                    }

                    let inv_denom = 1.0 / denom;
                    for &j in placed {
                        scores[j] *= inv_denom;
                    }

                    for dk in 0..GT_D_K {
                        let mut head_val = 0.0f32;
                        for &j in placed {
                            head_val += scores[j] * $v[head][j][dk];
                        }
                        let out_col = head * GT_D_K + dk;
                        for d_out in 0..GT_D_MODEL {
                            attn_out[i][d_out] += head_val * $wo[i][out_col][d_out];
                        }
                    }
                }
            }

            // Residual + Bias + LayerNorm 1 (only placed pieces)
            for &tok in placed {
                for d in 0..GT_D_MODEL {
                    $layer_h[tok][d] += attn_out[tok][d] + $bo[tok][d];
                }
                layer_norm(&mut $layer_h[tok], &$ng1[tok], &$nb1[tok]);
            }

            // FFN + Residual + LayerNorm 2 (only placed pieces)
            for &tok in placed {
                let mut ffn1 = [0.0f32; GT_D_FF];
                for f in 0..GT_D_FF {
                    let mut sum = $fb1[tok][f];
                    for d in 0..GT_D_MODEL {
                        sum += $layer_h[tok][d] * $fw1[tok][d][f];
                    }
                    ffn1[f] = relu(sum);
                }

                for d in 0..GT_D_MODEL {
                    let mut sum = $fb2[tok][d];
                    for f in 0..GT_D_FF {
                        sum += ffn1[f] * $fw2[tok][f][d];
                    }
                    $layer_h[tok][d] += sum;
                }
                layer_norm(&mut $layer_h[tok], &$ng2[tok], &$nb2[tok]);
            }
        };
    }

    // Layer 0 (uses cached Q, K, V)
    run_attention_and_ffn!(
        h, cache.l0_q, cache.l0_k, cache.l0_v,
        GT_L0_WO, GT_L0_BO, GT_L0_EDGE_BIAS,
        GT_L0_NORM1_G, GT_L0_NORM1_B,
        GT_L0_FFN1_W, GT_L0_FFN1_B, GT_L0_FFN2_W, GT_L0_FFN2_B,
        GT_L0_NORM2_G, GT_L0_NORM2_B
    );

    // Layer 1: compute Q, K, V from Layer-0 output h (only placed pieces)
    let mut q1 = [[[0.0f32; GT_D_K]; GT_NUM_TOKENS]; GT_N_HEADS];
    let mut k1 = [[[0.0f32; GT_D_K]; GT_NUM_TOKENS]; GT_N_HEADS];
    let mut v1 = [[[0.0f32; GT_D_K]; GT_NUM_TOKENS]; GT_N_HEADS];

    for &tok in placed {
        for head in 0..GT_N_HEADS {
            for dk in 0..GT_D_K {
                let col = head * GT_D_K + dk;
                let mut sum_q = 0.0f32;
                let mut sum_k = 0.0f32;
                let mut sum_v = 0.0f32;
                for d in 0..GT_D_MODEL {
                    let val = h[tok][d];
                    sum_q += val * GT_L1_WQ[tok][d][col];
                    sum_k += val * GT_L1_WK[tok][d][col];
                    sum_v += val * GT_L1_WV[tok][d][col];
                }
                q1[head][tok][dk] = sum_q;
                k1[head][tok][dk] = sum_k;
                v1[head][tok][dk] = sum_v;
            }
        }
    }

    run_attention_and_ffn!(
        h, q1, k1, v1,
        GT_L1_WO, GT_L1_BO, GT_L1_EDGE_BIAS,
        GT_L1_NORM1_G, GT_L1_NORM1_B,
        GT_L1_FFN1_W, GT_L1_FFN1_B, GT_L1_FFN2_W, GT_L1_FFN2_B,
        GT_L1_NORM2_G, GT_L1_NORM2_B
    );

    // Flatten all 28 tokens
    const FLATTEN_DIM: usize = GT_NUM_TOKENS * GT_D_MODEL;
    let mut flat = [0.0f32; FLATTEN_DIM];
    for tok in 0..GT_NUM_TOKENS {
        for d in 0..GT_D_MODEL {
            flat[tok * GT_D_MODEL + d] = h[tok][d];
        }
    }

    // Head MLP: [FLATTEN_DIM -> 32 -> 16 -> 8 -> 1]
    let mut l1 = [0.0f32; 32];
    for j in 0..32 {
        let mut sum = GT_HEAD_B1[j];
        for i in 0..FLATTEN_DIM {
            sum += flat[i] * GT_HEAD_W1[i][j];
        }
        l1[j] = relu(sum);
    }

    let mut l2 = [0.0f32; 16];
    for j in 0..16 {
        let mut sum = GT_HEAD_B2[j];
        for i in 0..32 {
            sum += l1[i] * GT_HEAD_W2[i][j];
        }
        l2[j] = relu(sum);
    }

    let mut l3 = [0.0f32; 8];
    for j in 0..8 {
        let mut sum = GT_HEAD_B3[j];
        for i in 0..16 {
            sum += l2[i] * GT_HEAD_W3[i][j];
        }
        l3[j] = relu(sum);
    }

    let mut logit = GT_HEAD_B4[0];
    for i in 0..8 {
        logit += l3[i] * GT_HEAD_W4[i][0];
    }

    let output = (softsign(logit) * 1.5).clamp(-1.0, 1.0) * 6000.0;
    output.round() as Eval
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    #[test]
    fn test_gt_inference_runs() {
        let mut board = Board::new();
        let tg = board.get_token_graph();
        let eval = gt_inference(&tg.features, &tg.adj);
        assert!(eval.abs() <= 9000);
    }

    #[test]
    fn test_gt_inference_with_cache_parity() {
        let pos = "Base+MLP;InProgress;White[3];wS1;bA1 wS1-;wA1 -wS1;bA2 bA1\\";
        let mut board = Board::parse_game_string(pos).unwrap();
        let tg = board.get_token_graph();

        let eval_direct = gt_inference(&tg.features, &tg.adj);

        let mut cache = TokenCache::new();
        cache.update(&tg.features);
        let eval_cached = gt_inference_with_cache(&cache, &tg.adj);

        assert_eq!(eval_direct, eval_cached);
    }

    #[test]
    fn test_gt_inference_fast_with_moves() {
        let pos = "Base+MLP;InProgress;White[3];wS1;bA1 wS1-;wA1 -wS1;bA2 bA1\\";
        let mut board = Board::parse_game_string(pos).unwrap();
        let my_moves = board.generate_moves();

        let tg_slow = board.get_token_graph();
        let tg_fast = board.get_token_graph_fast(&my_moves);

        assert_eq!(tg_slow.features, tg_fast.features);
        assert_eq!(tg_slow.adj, tg_fast.adj);

        reset_token_cache();
        let eval_fast = gt_inference_fast(&tg_fast.features, &tg_fast.adj);
        let eval_slow = gt_inference(&tg_slow.features, &tg_slow.adj);
        assert_eq!(eval_fast, eval_slow);
    }

    #[test]
    fn test_cache_parity_random_walk_with_undo() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Persistent cache across evaluations
        let mut cache = TokenCache::new();

        for game_idx in 0..100 {
            let mut board = Board::new();

            for step in 0..30 {
                // Step 1: Generate moves and make move 1
                let moves1 = board.generate_moves();
                if moves1.is_empty() {
                    break;
                }
                let m1 = moves1[rng.random_range(0..moves1.len())];
                board.do_action(m1);

                // Intermediate Check 1 (after move 1)
                let moves_after_m1 = board.generate_moves();
                let tg1 = board.get_token_graph_fast(&moves_after_m1);
                let cached_eval1 = cache.update_and_infer(&tg1.features, &tg1.adj);
                let scratch_eval1 = gt_inference(&tg1.features, &tg1.adj);
                assert_eq!(
                    cached_eval1, scratch_eval1,
                    "Cache mismatch at game {} step {} after move 1",
                    game_idx, step
                );

                // Step 2: Generate moves and make move 2
                if moves_after_m1.is_empty() {
                    break;
                }
                let m2 = moves_after_m1[rng.random_range(0..moves_after_m1.len())];
                board.do_action(m2);

                // Intermediate Check 2 (after move 2)
                let moves_after_m2 = board.generate_moves();
                let tg2 = board.get_token_graph_fast(&moves_after_m2);
                let cached_eval2 = cache.update_and_infer(&tg2.features, &tg2.adj);
                let scratch_eval2 = gt_inference(&tg2.features, &tg2.adj);
                assert_eq!(
                    cached_eval2, scratch_eval2,
                    "Cache mismatch at game {} step {} after move 2",
                    game_idx, step
                );

                // Step 3: Undo move 2
                board.undo_action();

                // Intermediate Check 3 (after undoing move 2)
                let moves_after_undo = board.generate_moves();
                let tg_undo = board.get_token_graph_fast(&moves_after_undo);
                let cached_eval_undo = cache.update_and_infer(&tg_undo.features, &tg_undo.adj);
                let scratch_eval_undo = gt_inference(&tg_undo.features, &tg_undo.adj);
                assert_eq!(
                    cached_eval_undo, scratch_eval_undo,
                    "Cache mismatch at game {} step {} after undoing move 2",
                    game_idx, step
                );
            }
        }
    }
}
