use std::process::Command;

fn main() {
    let hash = Command::new("git")
        .args(&["describe", "--always", "--dirty"])
        .output()
        .unwrap();
    println!("cargo:rustc-env=GIT_HASH={}", String::from_utf8_lossy(&hash.stdout).trim());

    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs");
}
