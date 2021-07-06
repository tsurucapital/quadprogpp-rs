fn main() {
    cxx_build::bridge("src/lib.rs")
        .file("upstream/src/Array.cc")
        .file("upstream/src/QuadProg++.cc")
        .include("upstream/src")
        .flag("-std=c++14")
        .flag("-Wno-extra")
        .compile("libquadprog.a");
}
