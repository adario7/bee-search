use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use crate::{board::Board, eval::Eval};

/* dataset:
- bee-search 0.1.0-94e66fc-MLP (mlp4) @ depth 8
- 384k position, half human, half self play

Best epoch (by val loss): 484, val_total_loss: 0.038836

MLP on test set (using best-epoch weights) - evaluation output:
  RMSE: 1182.4079
  MAE:  755.5481
  R2:   0.8179

Note: 8-bit quantized weights (i8) and 32-bit fixed point biases (i32).
*/

// state & helper functions for mlp-bench
static ACCUMULATE_ENABLED: AtomicBool = AtomicBool::new(false);
static ACCUMULATED_X: Mutex<Vec<Vec<i16>>> = Mutex::new(Vec::new());

pub fn set_accumulation(enabled: bool) {
    ACCUMULATE_ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn clear_accumulated_features() {
    if let Ok(mut guard) = ACCUMULATED_X.lock() {
        guard.clear();
    }
}

pub fn take_accumulated_features() -> Vec<Vec<i16>> {
    ACCUMULATE_ENABLED.store(false, Ordering::Relaxed);
    if let Ok(mut guard) = ACCUMULATED_X.lock() {
        std::mem::take(&mut *guard)
    } else {
        Vec::new()
    }
}

pub fn mlp_inference(x: &[i16]) -> Eval {
    if ACCUMULATE_ENABLED.load(Ordering::Relaxed) {
        if let Ok(mut guard) = ACCUMULATED_X.lock() {
            guard.push(x.to_vec());
        }
    }

    let mut cat = [0i32; 12];
    shared_inference_a(&x[..Board::FN], &mut cat[..6]);
    shared_inference_b(&x[Board::FN..], &mut cat[6..]);
    head_inference(&cat)
}

fn layer(x: &[i32], w: &[i8], b: &[i32], y: &mut [i32], relu: bool, shift: u32) {
    debug_assert_eq!(w.len(), x.len() * y.len());
    debug_assert_eq!(b.len(), y.len());
    y.copy_from_slice(b);
    let y_len = y.len();
    for i in 0..x.len() {
        let xi = x[i];
        if xi == 0 { continue; }
        let w_row = &w[i * y_len..(i + 1) * y_len];
        for j in 0..y_len {
            y[j] += w_row[j] as i32 * xi;
        }
    }
    if relu {
        for yj in y.iter_mut() {
            *yj = (*yj).max(0);
        }
    }
    if shift > 0 {
        for yj in y.iter_mut() {
            *yj >>= shift;
        }
    }
}

fn shared_inference_a(x: &[i16], y: &mut [i32]) {
    let mut x0 = [0i32; Board::FN];
    for i in 0..Board::FN {
        x0[i] = x[i] as i32;
    }
    let mut x1 = [0i32; 10];
    layer(&x0, A_W01.as_flattened(), &A_B1, &mut x1, true, 0);
    let mut x2 = [0i32; 8];
    layer(&x1, A_W12.as_flattened(), &A_B2, &mut x2, true, 6);
    let mut x3 = [0i32; 6];
    layer(&x2, A_W23.as_flattened(), &A_B3, &mut x3, true, 6);
    y.copy_from_slice(&x3);
}

fn shared_inference_b(x: &[i16], y: &mut [i32]) {
    let mut x0 = [0i32; Board::FN];
    for i in 0..Board::FN {
        x0[i] = x[i] as i32;
    }
    let mut x1 = [0i32; 10];
    layer(&x0, B_W01.as_flattened(), &B_B1, &mut x1, true, 0);
    let mut x2 = [0i32; 8];
    layer(&x1, B_W12.as_flattened(), &B_B2, &mut x2, true, 6);
    let mut x3 = [0i32; 6];
    layer(&x2, B_W23.as_flattened(), &B_B3, &mut x3, true, 6);
    y.copy_from_slice(&x3);
}

fn softsign(x: f32) -> f32 {
    x / (1.0 + x.abs())
}

fn head_inference(x3: &[i32; 12]) -> Eval {
    let mut x4 = [0i32; 8];
    layer(x3, H_W34.as_flattened(), &H_B4, &mut x4, true, 6);
    let mut x5 = [0i32; 6];
    layer(&x4, H_W45.as_flattened(), &H_B5, &mut x5, true, 6);
    let mut x6 = [0i32; 1];
    layer(&x5, H_W56.as_flattened(), &H_B6, &mut x6, false, 0);
    
    let logit = x6[0] as f32 / 4096.0;
    let output = (softsign(logit) * 1.5).clamp(-1.0, 1.0) * 6000.0;
    output.round() as Eval
}



// --- half_a (current player) ---
const A_W01: [[i8; 10]; 52] = [
[-4i8, 1i8, -4i8, 0i8, 5i8, -27i8, 1i8, 20i8, 3i8, 57i8],
[-1i8, 7i8, -5i8, -5i8, -15i8, -43i8, 6i8, 24i8, 1i8, 64i8],
[5i8, -3i8, 2i8, -2i8, 23i8, -11i8, -2i8, -3i8, -21i8, 43i8],
[7i8, 8i8, 1i8, -14i8, -9i8, -19i8, -1i8, 12i8, -9i8, 56i8],
[38i8, 23i8, -16i8, 11i8, 8i8, 35i8, -42i8, -14i8, 14i8, -3i8],
[62i8, 7i8, 0i8, 17i8, 12i8, 44i8, 3i8, -12i8, 28i8, -12i8],
[-15i8, 23i8, -14i8, -8i8, -13i8, -41i8, 18i8, 26i8, 48i8, 6i8],
[10i8, -6i8, 0i8, 13i8, 38i8, 12i8, 22i8, -24i8, 7i8, -14i8],
[-1i8, -5i8, 0i8, 12i8, 14i8, 6i8, 1i8, -2i8, -22i8, -8i8],
[0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8],
[-6i8, -1i8, 2i8, 18i8, -2i8, 1i8, -10i8, -3i8, 6i8, 0i8],
[1i8, -1i8, 27i8, 20i8, 2i8, 1i8, -9i8, 2i8, -6i8, -4i8],
[-1i8, 2i8, 32i8, 7i8, 0i8, -3i8, -9i8, 2i8, -2i8, -3i8],
[1i8, -8i8, -2i8, 7i8, 6i8, 1i8, -2i8, -3i8, 1i8, -3i8],
[1i8, 2i8, 4i8, 9i8, 5i8, -2i8, 4i8, 2i8, 4i8, 15i8],
[-9i8, -5i8, 3i8, -9i8, 10i8, -1i8, 13i8, 3i8, 0i8, 10i8],
[-4i8, 15i8, 18i8, -18i8, -7i8, -15i8, -1i8, 10i8, -2i8, 3i8],
[-2i8, -7i8, 33i8, -13i8, 0i8, -15i8, 0i8, 4i8, -5i8, -1i8],
[-3i8, 0i8, 31i8, 2i8, 0i8, -15i8, 1i8, 1i8, 3i8, -4i8],
[0i8, -15i8, -2i8, 7i8, 9i8, 3i8, -4i8, -7i8, 3i8, -4i8],
[2i8, 3i8, -9i8, -5i8, -18i8, -12i8, -3i8, 7i8, 4i8, 27i8],
[-5i8, 6i8, -7i8, -6i8, -6i8, -7i8, 7i8, 3i8, 6i8, 23i8],
[-4i8, -44i8, -3i8, 10i8, -2i8, 12i8, -29i8, 5i8, 10i8, -3i8],
[0i8, 6i8, 23i8, 19i8, 0i8, 6i8, -2i8, -3i8, 0i8, -5i8],
[-2i8, 0i8, 27i8, 16i8, 9i8, -4i8, -1i8, -4i8, 7i8, -11i8],
[0i8, 2i8, 0i8, 1i8, 1i8, 1i8, 1i8, -1i8, 0i8, 0i8],
[5i8, 6i8, 0i8, -2i8, -7i8, -1i8, 3i8, 4i8, -7i8, 19i8],
[-15i8, 3i8, 12i8, -1i8, -9i8, 1i8, -1i8, 6i8, -2i8, 23i8],
[-9i8, 2i8, 1i8, -2i8, -14i8, 21i8, -2i8, 9i8, -3i8, 8i8],
[-3i8, -3i8, 28i8, 10i8, -1i8, 9i8, -2i8, 9i8, -13i8, -4i8],
[-1i8, -2i8, 24i8, 22i8, 0i8, 13i8, -8i8, -4i8, 14i8, -3i8],
[0i8, -3i8, -3i8, 9i8, 3i8, 0i8, -2i8, -7i8, 0i8, -3i8],
[-16i8, -2i8, 0i8, -10i8, -11i8, -39i8, 4i8, 19i8, 12i8, 23i8],
[-7i8, -2i8, -1i8, -9i8, -5i8, -10i8, -13i8, -13i8, -19i8, 14i8],
[-10i8, 29i8, -11i8, 33i8, 14i8, 22i8, -19i8, 11i8, 12i8, -7i8],
[-8i8, -13i8, 16i8, 10i8, 8i8, 11i8, -15i8, -6i8, -3i8, -7i8],
[-2i8, -2i8, 22i8, 6i8, 2i8, -3i8, -6i8, -6i8, 2i8, -3i8],
[0i8, -9i8, 0i8, 0i8, 1i8, 0i8, 1i8, -7i8, 0i8, 0i8],
[2i8, 6i8, 0i8, -8i8, -1i8, -13i8, 1i8, 7i8, -1i8, 17i8],
[-6i8, 5i8, 1i8, 4i8, 15i8, 0i8, -14i8, -6i8, -12i8, 4i8],
[-3i8, 8i8, -8i8, 17i8, 3i8, 8i8, -19i8, -2i8, 9i8, -2i8],
[0i8, -6i8, 21i8, 14i8, 7i8, 10i8, -7i8, 1i8, -5i8, -5i8],
[1i8, 2i8, 29i8, 8i8, -9i8, -1i8, -9i8, 0i8, -11i8, 1i8],
[0i8, -3i8, 0i8, 3i8, 4i8, 0i8, 2i8, -2i8, 1i8, -1i8],
[1i8, -1i8, 5i8, 5i8, 16i8, 6i8, 4i8, 3i8, 5i8, 10i8],
[-8i8, -1i8, -2i8, 4i8, 17i8, 0i8, -1i8, 3i8, 0i8, 5i8],
[-17i8, 3i8, -5i8, 7i8, 15i8, 19i8, -32i8, 0i8, -3i8, -11i8],
[-1i8, -14i8, 10i8, 18i8, 33i8, 22i8, 20i8, 3i8, -10i8, -23i8],
[-3i8, 5i8, 24i8, 11i8, -11i8, 1i8, -11i8, 5i8, 6i8, 2i8],
[3i8, -7i8, -2i8, 3i8, 7i8, 1i8, 20i8, -1i8, 0i8, -4i8],
[5i8, -3i8, -6i8, -3i8, -8i8, -13i8, -4i8, 5i8, 11i8, 24i8],
[67i8, 0i8, -1i8, 1i8, -8i8, -14i8, 7i8, 13i8, -3i8, 21i8]
];

const A_B1: [i32; 10] = [-58i32, -2i32, -30i32, -29i32, 7i32, 64i32, 10i32, -10i32, 27i32, -74i32];

const A_W12: [[i8; 8]; 10] = [
[0i8, -4i8, -38i8, 14i8, 10i8, 19i8, 8i8, -44i8],
[1i8, 3i8, -2i8, 18i8, -9i8, -14i8, -4i8, -1i8],
[-2i8, 2i8, -17i8, -10i8, 26i8, -8i8, -5i8, 2i8],
[2i8, 2i8, 2i8, -23i8, 18i8, 17i8, -5i8, 5i8],
[21i8, 6i8, 0i8, 11i8, 6i8, 7i8, -5i8, -5i8],
[-12i8, -20i8, -9i8, 1i8, -14i8, 14i8, 37i8, 11i8],
[0i8, -1i8, 4i8, -1i8, -10i8, -17i8, -5i8, -1i8],
[5i8, 14i8, 2i8, 8i8, 8i8, -8i8, -6i8, -6i8],
[-6i8, 28i8, -8i8, 2i8, 0i8, 3i8, 0i8, 7i8],
[27i8, -6i8, 8i8, -13i8, -7i8, -7i8, -5i8, 24i8]
];

const A_B2: [i32; 8] = [-6033i32, 2212i32, 2134i32, 2677i32, -2990i32, 1793i32, -1046i32, -2871i32];

const A_W23: [[i8; 6]; 8] = [
[-6i8, -2i8, -25i8, -10i8, 87i8, 0i8],
[33i8, -12i8, 4i8, -12i8, -11i8, -3i8],
[-9i8, 8i8, 0i8, 19i8, -20i8, -13i8],
[-23i8, -2i8, 12i8, -11i8, -14i8, -25i8],
[-26i8, 29i8, -9i8, 24i8, -4i8, -14i8],
[2i8, -9i8, 1i8, 9i8, -5i8, 32i8],
[-3i8, 5i8, -22i8, 4i8, 1i8, 14i8],
[5i8, -13i8, -26i8, 9i8, 34i8, -17i8]
];

const A_B3: [i32; 6] = [732i32, -76i32, 1715i32, -763i32, -3062i32, 1991i32];

// --- half_b (opponent) ---
const B_W01: [[i8; 10]; 52] = [
[7i8, 3i8, 2i8, 3i8, -26i8, 0i8, 6i8, 13i8, -10i8, 57i8],
[15i8, 15i8, 4i8, -14i8, -42i8, 2i8, 14i8, 19i8, -12i8, 63i8],
[-11i8, -17i8, 10i8, 16i8, 4i8, 3i8, 24i8, 4i8, -44i8, 45i8],
[-1i8, 6i8, 4i8, -17i8, -14i8, 3i8, 7i8, 10i8, -28i8, 59i8],
[-21i8, 36i8, -17i8, 9i8, 35i8, 15i8, -16i8, -35i8, 6i8, -3i8],
[9i8, 25i8, -5i8, 25i8, 42i8, -13i8, 6i8, -70i8, 18i8, 13i8],
[57i8, 16i8, -16i8, 4i8, -43i8, -10i8, -30i8, 12i8, -10i8, 3i8],
[2i8, 0i8, -3i8, 13i8, 8i8, -19i8, 9i8, -25i8, 10i8, -7i8],
[-6i8, -11i8, 8i8, 14i8, 14i8, -5i8, 11i8, 1i8, 1i8, -4i8],
[0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8],
[-6i8, 12i8, 5i8, 8i8, 0i8, 2i8, -3i8, 9i8, 14i8, -1i8],
[-5i8, 5i8, 31i8, 14i8, 6i8, 6i8, 5i8, 5i8, 8i8, -6i8],
[-5i8, -3i8, 32i8, 5i8, -1i8, 0i8, -8i8, 0i8, 4i8, -4i8],
[-2i8, 1i8, -1i8, 11i8, 1i8, -1i8, 3i8, 1i8, 0i8, -4i8],
[5i8, 5i8, 4i8, 10i8, -3i8, 0i8, 1i8, 5i8, -4i8, 15i8],
[13i8, -14i8, 1i8, -2i8, -4i8, -10i8, 2i8, 4i8, 0i8, 14i8],
[8i8, -3i8, 13i8, -19i8, -17i8, 18i8, 6i8, -2i8, 6i8, 1i8],
[7i8, -5i8, 33i8, -9i8, -13i8, 6i8, 14i8, -6i8, 4i8, -2i8],
[8i8, -4i8, 30i8, 2i8, -13i8, -4i8, 3i8, -6i8, -1i8, -2i8],
[-3i8, 0i8, 0i8, 12i8, 4i8, -10i8, 2i8, 0i8, 0i8, -5i8],
[5i8, 14i8, -8i8, -12i8, -14i8, 4i8, 1i8, 12i8, -1i8, 26i8],
[9i8, 2i8, -9i8, -8i8, -12i8, -5i8, -5i8, 10i8, 0i8, 21i8],
[-17i8, 5i8, -3i8, 18i8, 12i8, 13i8, -19i8, 5i8, 3i8, -6i8],
[-2i8, 3i8, 26i8, 13i8, 8i8, 5i8, -2i8, 3i8, 3i8, -5i8],
[-3i8, -2i8, 25i8, 17i8, -5i8, 0i8, -1i8, -6i8, 6i8, -11i8],
[0i8, 0i8, 0i8, 1i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8],
[3i8, 9i8, 1i8, -5i8, 2i8, 2i8, 9i8, 7i8, 1i8, 20i8],
[-4i8, -10i8, 7i8, 0i8, -5i8, -1i8, -18i8, 15i8, -18i8, 26i8],
[4i8, 2i8, -2i8, -7i8, 21i8, 10i8, -20i8, 18i8, 12i8, 8i8],
[4i8, -6i8, 33i8, 2i8, 17i8, 4i8, 1i8, 7i8, 10i8, -4i8],
[-7i8, 7i8, 20i8, 15i8, 12i8, -5i8, -28i8, 1i8, 5i8, 0i8],
[-6i8, 2i8, 0i8, 7i8, 1i8, -4i8, 4i8, 1i8, 1i8, -3i8],
[5i8, -10i8, 6i8, -18i8, -45i8, 2i8, -8i8, 3i8, -3i8, 25i8],
[-21i8, -15i8, 4i8, -7i8, -3i8, 3i8, 11i8, 11i8, -1i8, 16i8],
[-22i8, 7i8, -12i8, 39i8, 18i8, 41i8, -7i8, 10i8, 14i8, -12i8],
[-9i8, -6i8, 17i8, 13i8, 13i8, 4i8, -1i8, 3i8, 10i8, -7i8],
[-4i8, 1i8, 24i8, -1i8, -2i8, -6i8, -11i8, -6i8, 3i8, -2i8],
[0i8, 0i8, 0i8, -1i8, 0i8, -9i8, 0i8, 0i8, 0i8, 0i8],
[5i8, 1i8, 2i8, -5i8, -10i8, 3i8, 7i8, 2i8, -5i8, 17i8],
[-11i8, -10i8, 6i8, 16i8, 6i8, 19i8, 14i8, -3i8, -2i8, 5i8],
[-9i8, 12i8, -10i8, 15i8, 7i8, 14i8, -10i8, 3i8, 9i8, -4i8],
[-5i8, 0i8, 22i8, 15i8, 13i8, 2i8, 1i8, -2i8, 5i8, -6i8],
[-12i8, 0i8, 30i8, -3i8, 4i8, 4i8, -3i8, 4i8, 5i8, 2i8],
[0i8, 0i8, 0i8, 4i8, 0i8, -3i8, 2i8, 0i8, -1i8, -1i8],
[8i8, -2i8, 2i8, 11i8, 7i8, 1i8, 4i8, -2i8, 0i8, 10i8],
[-2i8, -12i8, -1i8, 14i8, 0i8, 1i8, 4i8, 0i8, -4i8, 5i8],
[-6i8, -8i8, -6i8, 21i8, 23i8, 41i8, -2i8, 0i8, 12i8, -14i8],
[10i8, -17i8, 12i8, 17i8, 30i8, -20i8, 14i8, -16i8, 2i8, -17i8],
[2i8, 10i8, 22i8, -1i8, 6i8, 4i8, -17i8, 8i8, -8i8, 1i8],
[6i8, -1i8, -2i8, -4i8, -1i8, -29i8, 5i8, -3i8, 3i8, 1i8],
[2i8, 11i8, -6i8, -1i8, -16i8, -2i8, -2i8, 6i8, -6i8, 21i8],
[9i8, 49i8, 8i8, -13i8, 8i8, 1i8, 26i8, -24i8, -48i8, 21i8]
];

const B_B1: [i32; 10] = [13i32, -12i32, -30i32, -27i32, 49i32, 24i32, 8i32, -7i32, 14i32, -86i32];

const B_W12: [[i8; 8]; 10] = [
[-10i8, -16i8, 0i8, -9i8, 15i8, -3i8, -4i8, -2i8],
[-14i8, -7i8, -12i8, -1i8, -5i8, 23i8, -6i8, 9i8],
[8i8, -4i8, -17i8, 18i8, 2i8, 0i8, 33i8, 9i8],
[14i8, 4i8, -15i8, -12i8, -6i8, -2i8, 4i8, -10i8],
[11i8, 31i8, 6i8, 4i8, 6i8, 12i8, -1i8, 25i8],
[-4i8, -8i8, 7i8, 3i8, 16i8, -8i8, -8i8, 11i8],
[18i8, -4i8, 3i8, 18i8, -2i8, 13i8, -4i8, -18i8],
[-4i8, -6i8, -2i8, 16i8, -5i8, -31i8, -6i8, 6i8],
[1i8, 12i8, -3i8, -2i8, -13i8, -23i8, 6i8, 1i8],
[32i8, -4i8, 5i8, -11i8, -12i8, 6i8, -10i8, 27i8]
];

const B_B2: [i32; 8] = [-4351i32, 4219i32, 3756i32, 3148i32, 1614i32, -2798i32, -1607i32, -1533i32];

const B_W23: [[i8; 6]; 8] = [
[-8i8, 3i8, 0i8, -4i8, 77i8, -5i8],
[36i8, -1i8, -10i8, 7i8, -45i8, -4i8],
[-16i8, 20i8, 1i8, -8i8, -12i8, -26i8],
[2i8, 3i8, 14i8, 3i8, -42i8, -18i8],
[-11i8, 19i8, -30i8, -5i8, -1i8, -2i8],
[42i8, 3i8, 12i8, 37i8, 6i8, -3i8],
[-11i8, -8i8, -7i8, 25i8, -1i8, -8i8],
[-12i8, -26i8, -20i8, -1i8, 20i8, -5i8]
];

const B_B3: [i32; 6] = [1774i32, -23i32, 2284i32, -1955i32, -4185i32, 3569i32];

// --- head ---
const H_W34: [[i8; 8]; 12] = [
[19i8, 0i8, -5i8, -22i8, 18i8, 0i8, -18i8, 24i8],
[19i8, 0i8, 5i8, -1i8, -10i8, 0i8, -9i8, -16i8],
[-16i8, 0i8, 44i8, -4i8, 6i8, 0i8, 19i8, 0i8],
[17i8, 0i8, 7i8, 10i8, 6i8, 0i8, -17i8, -1i8],
[-78i8, 0i8, -23i8, 14i8, -27i8, 0i8, -34i8, -49i8],
[-35i8, 0i8, 5i8, -21i8, 19i8, 0i8, 33i8, 20i8],
[15i8, 0i8, 20i8, 25i8, -43i8, 0i8, -14i8, -34i8],
[-23i8, 0i8, -6i8, 14i8, 16i8, 0i8, -6i8, 5i8],
[13i8, 0i8, -23i8, 7i8, -16i8, 0i8, 15i8, -9i8],
[-7i8, 0i8, -4i8, -24i8, 18i8, 0i8, 19i8, 33i8],
[-59i8, 0i8, -27i8, 2i8, -51i8, 0i8, -61i8, -71i8],
[34i8, 0i8, -35i8, -2i8, 14i8, 0i8, -17i8, 19i8]
];

const H_B4: [i32; 8] = [809i32, 0i32, -650i32, -853i32, -320i32, 0i32, 1527i32, 533i32];

const H_W45: [[i8; 6]; 8] = [
[0i8, 38i8, -23i8, 65i8, -24i8, 59i8],
[0i8, 0i8, 0i8, 0i8, 0i8, 0i8],
[0i8, 29i8, -7i8, -2i8, -33i8, 7i8],
[0i8, -11i8, -11i8, -19i8, 21i8, -21i8],
[0i8, -24i8, 60i8, 11i8, 33i8, -5i8],
[0i8, 0i8, 0i8, 0i8, 0i8, 0i8],
[0i8, -24i8, -20i8, -47i8, 38i8, -22i8],
[0i8, -23i8, 59i8, -19i8, 54i8, 4i8]
];

const H_B5: [i32; 6] = [0i32, 2202i32, 428i32, -733i32, 1163i32, -309i32];

const H_W56: [[i8; 1]; 6] = [
[0i8],
[-25i8],
[81i8],
[-80i8],
[35i8],
[-53i8]
];

const H_B6: [i32; 1] = [-342i32];
