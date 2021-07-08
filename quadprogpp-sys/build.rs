fn main() {
    let mut build = cxx_build::bridge("src/lib.rs");
    build
        .file("upstream/src/Array.cc")
        .file("upstream/src/QuadProg++.cc")
        .include("upstream/src")
        .flag("-std=c++14")
        .flag("-Wno-extra");
    if cfg!(features = "trace-solver") {
        build.define("TRACE_SOLVER", "1");
    }
    build.compile("libquadprog.a");
}
