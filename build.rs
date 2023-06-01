use std::{path::Path, env::var_os};

fn main() {
    //no power9, windows later, mac later
    let mut cflags: Vec<(u8, String)> = Vec::new();
    let mut ldflags: Vec<(u8, String)> = Vec::new();
    let mut objs: Vec<&str> = Vec::new();
    cflags.push((1, "-O3".to_owned()));
    cflags.push((1, "-Wall".to_owned()));
    cflags.push((1, "-Wextra".to_owned()));
    cflags.push((1, "-Wpedantic".to_owned()));
    cflags.push((1, "-Wcast-qual".to_owned()));
    let mut cxxflags: Vec<(u8, String)> = cflags.clone();
    cflags.push((1, "-std=c11".to_owned()));
    cxxflags.push((1, "-std=c++11".to_owned()));
    cxxflags.push((1, "-Wno-unused-function".to_owned()));
    cxxflags.push((1, "-Wno-multichar".to_owned()));
    cflags.push((1, "-Wdouble-promotion".to_owned()));
    cflags.push((1, "-Wshadow".to_owned()));
    cflags.push((1, "-Wstrict-prototypes".to_owned()));
    cflags.push((1, "-Wpointer-arith".to_owned()));
    if cfg!(unix) {
        cflags.push((1, "-pthread".to_owned()));
        cxxflags.push((1, "-pthread".to_owned()));
    }
    if cfg!(any(target_arch = "x86_64", target_arch = "x86")) {
        cflags.push((1, "-march=native".to_owned()));
        cflags.push((1, "-mtune=native".to_owned()));
        cflags.push((1, "-mfma".to_owned()));
        cflags.push((1, "-mf16c".to_owned()));
        cflags.push((1, "-mavx".to_owned()));
        cxxflags.push((1, "-march=native".to_owned()));
        cxxflags.push((1, "-mtune=native".to_owned()));
        cxxflags.push((1, "-mfma".to_owned()));
        cxxflags.push((1, "-mf16c".to_owned()));
        cxxflags.push((1, "-mavx".to_owned()));
    }
    if option_env!("LLAMA_OPENBLAS").is_some() && cfg!(unix) {
        cflags.push((2, "GGML_USE_OPENBLAS".to_owned()));
        cflags.push((3, "/usr/local/include/openblas".to_owned()));
        cflags.push((3, "/usr/include/openblas".to_owned()));
        ldflags.push((1, "openblas".to_owned()));
        ldflags.push((1, "cblas".to_owned())); //possible error on arch
    }
    if option_env!("LLAMA_BLIS").is_some() && cfg!(unix) {
        cflags.push((2, "GGML_USE_OPENBLAS".to_owned()));
        cflags.push((3, "/usr/local/include/blis".to_owned()));
        cflags.push((3, "/usr/include/blis".to_owned()));
        ldflags.push((2, "blis".to_owned()));
        ldflags.push((2, "/usr/local/lib".to_owned()))
    }
    if option_env!("LLAMA_CUBLAS").is_some() && cfg!(unix) {
        cflags.push((2, "GGML_USE_CUBLAS".to_owned()));
        cflags.push((3, "/usr/local/cuda/include".to_owned()));
        cflags.push((3, "/opt/cuda/include".to_owned()));
        cxxflags.push((2, "GGML_USE_CUBLAS".to_owned()));
        cxxflags.push((3, "/usr/local/cuda/include".to_owned()));
        cxxflags.push((3, "/opt/cuda/include".to_owned()));
        ldflags.push((1, "cublas".to_owned()));
        ldflags.push((1, "culibos".to_owned()));
        ldflags.push((1, "cudart".to_owned()));
        ldflags.push((1, "cublasLt".to_owned()));
        ldflags.push((1, "pthread".to_owned()));
        ldflags.push((1, "dl".to_owned()));
        ldflags.push((1, "rt".to_owned()));
        ldflags.push((2, "/usr/local/cuda/lib64".to_owned()));
        ldflags.push((2, "/opt/cuda/lib64".to_owned()));
        if let Some(cuda_path) = option_env!("CUDA_PATH") {
            cflags.push((3, (cuda_path.to_owned() + "/targets/x86_64-linux/include")));
            cxxflags.push((3, (cuda_path.to_owned() + "/targets/x86_64-linux/include")));
            ldflags.push((2, (cuda_path.to_owned() + "/targets/x86_64-linux/lib")));
        }
        objs.push("ggml-cuda.o");
        let ggml_cuda_dmmv_x = if let Some(val) = option_env!("LLAMA_CUDA_DMMV_X") {
            val
        } else {
            "32"
        };
        let ggml_cuda_dmmv_y = if let Some(val) = option_env!("LLAMA_CUDA_DMMV_Y") {
            val
        } else {
            "1"
        };
        let build = &mut cc::Build::new();
        let cuda = build
            .cuda(true)
            .flag("--std=c++14")
            .flag("--forward-unknown-to-host-compiler")
            .flag("-arch=native")
            .file("ggml-cuda.cu")
            .include(".")
            .define("GGML_CUDA_DMMV_X", ggml_cuda_dmmv_x)
            .define("GGML_CUDA_DMMV_Y", ggml_cuda_dmmv_y);
        for arg in cxxflags.as_slice() {
            if arg.0 == 1 {
                cuda.flag_if_supported(&arg.1);
            } else if arg.0 == 2 {
                cuda.define(&arg.1, None);
            } else {
                cuda.include(Path::new(&arg.1));
            }
        }
        cuda.flag("-Wno-pedantic").compile("ggml-cuda");
        println!("cargo:rerun-if-changed=ggml-cuda.cu");
    }
    if option_env!("LLAMA_CLBLAST").is_some() && cfg!(unix) {
        cflags.push((2, "GGML_USE_CLBLAST".to_owned()));
        cxxflags.push((2, "GGML_USE_CLBLAST".to_owned()));
        ldflags.push((1, "clblast".to_owned()));
        ldflags.push((1, "OpenCL".to_owned()));
        objs.push("ggml-opencl.o");
        let build = &mut cc::Build::new();
        let opencl = build.cpp(true).file("ggml-opencl.cpp").include(".");
        for arg in cxxflags.as_slice() {
            if arg.0 == 1 {
                opencl.flag_if_supported(&arg.1);
            } else if arg.0 == 2 {
                opencl.define(&arg.1, None);
            } else {
                opencl.include(Path::new(&arg.1));
            }
        }
        opencl.compile("ggml-opencl");
        println!("cargo:rerun-if-changed=ggml-opencl.cpp");
    }
    if cfg!(target_arch="aarch64") {
        cflags.push((1, "-mcpu=native".to_owned()));
        cxxflags.push((1, "-mcpu=native".to_owned()));
    }
    if cfg!(target_arch="arm") { // combine all arm params, hope they work
        cflags.push((1, "-mfpu=neon-fp-armv8".to_owned()));
        cflags.push((1, "-mfp16-format=ieee".to_owned()));
        cflags.push((1, "-mno-unaligned-access".to_owned()));
        cflags.push((1, "-funsafe-math-optimizations".to_owned()));
    }
    let build = &mut cc::Build::new();
    let ggml = build;
    ggml.file("ggml.c").include(".");
    for arg in cflags.as_slice() {
        if arg.0 == 1 {
            ggml.flag_if_supported(&arg.1);
        }
        else if arg.0 == 2 {
            ggml.define(&arg.1, None);
        }
        else {
            ggml.include(Path::new(&arg.1));
        }
    }
    ggml.compile("ggml");
    let build = &mut cc::Build::new();
    let llama = build;
    llama.cpp(true).file("llama.cpp").include(".");
    for arg in cxxflags.as_slice() {
        if arg.0 == 1 {
            llama.flag_if_supported(&arg.1);
        }
        else if arg.0 == 2 {
            llama.define(&arg.1, None);
        }
        else {
            llama.include(Path::new(&arg.1));
        }
    }
    llama.compile("llama");
    let out_dir = &var_os("OUT_DIR").unwrap();
    let build = &mut cc::Build::new();
    let libllama = build;
    libllama.cpp(true)
        .object(Path::new(out_dir).join("llama.o"))
        .object(Path::new(out_dir).join("ggml.o"));
    for obj in objs {
        libllama.object(Path::new(out_dir).join(obj));
    }
    for arg in cxxflags.as_slice() {
        if arg.0 == 1 {
            libllama.flag_if_supported(&arg.1);
        }
        else if arg.0 == 2 {
            libllama.define(&arg.1, None);
        }
        else {
            libllama.include(Path::new(&arg.1));
        }
    }
    libllama.shared_flag(true).pic(true);
    for arg in ldflags.as_slice() {
        if arg.0 == 1 {
            libllama.flag(&("-l".to_owned() + &arg.1));
        }
        else {
            libllama.flag(&("-L".to_owned() + Path::new(&arg.1).to_str().unwrap()));
        }
    }
    libllama.compile("libllama.so");
    for lib in ldflags.as_slice() {
        if lib.0 == 1 {
            println!("cargo:rustc-link-lib={}", lib.1);
        }
        else {
            println!("cargo:rustc-link-search={}", lib.1);
        }
    }
    println!("cargo:rerun-if-changed=llama.cpp");
    println!("cargo:rerun-if-changed=ggml.c");
    println!("cargo:rerun-if-changed=wrapper.h");
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");
    bindings.write_to_file(Path::new(out_dir).join("bindings.rs"))
        .expect("Unable to write bindings");
    //todo: add configuration options for gpu offload enable
}
