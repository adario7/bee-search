macro_rules! perf_tune {
    ($name:ident, $ty:ty, $default:expr, $current:expr) => {
        ::paste::paste! {
            pub const $name: $ty = $current;
            pub const [<$name _DEFAULT>]: $ty = $default;
            pub const [<$name _STR>]: &'static str = {
                if $name == [<$name _DEFAULT>] {
                    ""
                } else {
                    concat!(stringify!($name), "=", stringify!($current))
                }
            };
        }
    };
}

pub const TUNING_STRS: &'static [&'static str] = &[
    RED_C_STR,
    RED_B_STR,
    NTR_P_STR,
    NTR_C_STR,
    RZO_B_STR,
    RZO_C_STR,
    FTP_B_STR,
    FTP_C_STR,
    NMP_T_STR,
    NMP_D_STR,
    ASP_W_STR,
];

perf_tune!(RED_C, f32, 0.68, 0.68);
perf_tune!(RED_B, f32, 1.0, 1.0);

perf_tune!(NTR_P, f32, 1.5, 1.5);
perf_tune!(NTR_C, f32, 2.8, 2.8);

perf_tune!(RZO_B, i32, 500, 500);
perf_tune!(RZO_C, i32, 200, 200);

perf_tune!(FTP_B, i32, 150, 150);
perf_tune!(FTP_C, i32, 120, 120);

perf_tune!(NMP_T, i16, 50, 50);
perf_tune!(NMP_D, u8, 2, 2);

perf_tune!(ASP_W, i64, 30, 30);
