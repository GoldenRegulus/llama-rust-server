use clap::Parser;
use libc::{c_char, c_float, c_int, size_t};
use rocket::{
    http::Status,
    log::private::{log, Level},
    response::status,
    serde::json::Json,
    tokio::{
        fs::read_dir,
        sync::{mpsc, oneshot, Mutex, RwLock},
    },
};
use serde::{Deserialize, Serialize};
use std::{ffi::CString, mem::size_of, path::PathBuf, sync::Arc};

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

struct LlamaContextPtr(*mut llama_context);

unsafe impl Send for LlamaContextPtr {}

fn init() {
    unsafe { llama_init_backend() };
}

fn load_model(
    path_to_model: &str,
    context_size: Option<i32>,
    gpu_offload: Option<i32>,
    seed: Option<i32>,
    kv_in_f16: Option<bool>,
    pin_memory: Option<bool>,
    no_swap: Option<bool>,
) -> Result<*mut llama_context, ()> {
    let mut params = unsafe { llama_context_default_params() };
    params.n_ctx = context_size.unwrap_or(params.n_ctx);
    params.n_gpu_layers = gpu_offload.unwrap_or(params.n_gpu_layers);
    params.seed = seed.unwrap_or(params.seed);
    params.f16_kv = kv_in_f16.unwrap_or(params.f16_kv);
    params.use_mmap = pin_memory.unwrap_or(params.use_mmap);
    params.use_mlock = no_swap.unwrap_or(params.use_mlock);
    let ctx = unsafe {
        llama_init_from_file(
            CString::new(path_to_model)
                .expect("Path contains invalid characters")
                .as_c_str()
                .as_ptr(),
            params,
        )
    };
    if ctx.is_null() {
        log!(Level::Error, "Unable to load model: Error during loading");
        return Err(());
    }
    //todo: lora adapter
    Ok(ctx)
}

fn free_memory(ctx: *mut llama_context) {
    unsafe { llama_free(ctx) };
    drop(ctx);
}

//todo: load session

fn tokenize_text(
    ctx: *mut llama_context,
    text: &str,
    add_beginning_of_sentence_token: bool,
) -> Vec<llama_token> {
    let mut res: Vec<llama_token> = Vec::new();
    let n = text.len()
        + (if add_beginning_of_sentence_token {
            size_of::<c_char>()
        } else {
            0
        });
    res.reserve(n);
    let n = unsafe {
        llama_tokenize(
            ctx,
            CString::new(text)
                .expect("Invalid characters")
                .as_c_str()
                .as_ptr(),
            res.as_mut_ptr(),
            n as c_int,
            add_beginning_of_sentence_token,
        )
    };
    assert!(n >= 0, "Could not tokenize input");
    res.truncate(n as usize);
    return res;
}

//flow: init, load, get input, tokenize, predict, untokenize, stream
enum ProcessState {
    WORKING,
    ERROR,
    OK,
}

#[derive(Clone)]
struct LoadParams {
    path_to_model_dir: PathBuf,
    context_size: Option<i32>,
    gpu_offload: Option<i32>,
    seed: Option<i32>,
    kv_in_f16: Option<bool>,
    pin_memory: Option<bool>,
    no_swap: Option<bool>,
}

struct MainState {
    process_state: ProcessState,
    has_output: bool,
    current_token: String,
    input_text: String,
    load_params: LoadParams,
    ctx: Option<Arc<Mutex<LlamaContextPtr>>>,
    current_model: Option<String>,
}

#[derive(Serialize)]
struct StatusResponse {
    status: &'static str,
}

#[rocket::get("/status")]
fn server_status(state: &rocket::State<MainState>) -> status::Accepted<Json<StatusResponse>> {
    match state.process_state {
        ProcessState::WORKING => status::Accepted(Some(Json(StatusResponse { status: "working" }))),
        ProcessState::ERROR => status::Accepted(Some(Json(StatusResponse { status: "error" }))),
        ProcessState::OK => status::Accepted(Some(Json(StatusResponse { status: "ok" }))),
    }
}

#[derive(Deserialize)]
#[serde(tag = "event")]
#[serde(rename_all = "lowercase")]
enum ModelEventRequest {
    LOAD { message: String },
    UNLOAD,
    LIST,
    CURRENT,
}

#[derive(Serialize)]
#[serde(tag = "response")]
#[serde(rename_all = "lowercase")]
enum ModelEventResponse {
    OK {
        #[serde(skip_serializing_if = "Option::is_none")]
        message: Option<String>,
    },
    #[serde(rename(serialize = "ok"))]
    OKVec { message: Vec<String> },
    ERROR {
        #[serde(skip_serializing_if = "Option::is_none")]
        message: Option<String>,
    },
}

#[derive(Serialize)]
#[serde(tag = "response")]
#[serde(rename_all = "lowercase")]
enum ModelEventResponseArray {
    OK { message: Vec<String> },
}

#[rocket::get("/models", data = "<user_input>")]
async fn change_model(
    state: &rocket::State<RwLock<MainState>>,
    user_input: Json<ModelEventRequest>,
) -> Result<status::Accepted<Json<ModelEventResponse>>, status::Custom<Json<ModelEventResponse>>> {
    match user_input.0 {
        ModelEventRequest::LOAD { message } => {
            let model_names =
                read_model_dir(&state.read().await.load_params.path_to_model_dir).await;
            if model_names.contains(&message) {
                let (sender, recv) = oneshot::channel::<Result<Arc<Mutex<LlamaContextPtr>>, ()>>();
                let params = state.read().await.load_params.clone();
                let msg = message.clone();
                let wrapped_ctx = state.read().await.ctx.clone();
                rocket::tokio::spawn(async move {
                    if let Some(existing_wrapped_ctx) = wrapped_ctx {
                        free_memory(existing_wrapped_ctx.lock().await.0);
                    }
                    let ctx: Result<*mut llama_context, ()> = load_model(
                        params.path_to_model_dir.join(msg).to_str().unwrap(),
                        params.context_size,
                        params.gpu_offload,
                        params.seed,
                        params.kv_in_f16,
                        params.pin_memory,
                        params.no_swap,
                    );
                    match ctx {
                        Ok(v) => {
                            sender
                                .send(Ok(Arc::new(Mutex::new(LlamaContextPtr(v)))))
                                .ok();
                        }
                        Err(_) => {
                            sender.send(Err(())).ok();
                        }
                    }
                });
                match recv.await {
                    Ok(v) => match v {
                        Ok(wrapped_ctx) => {
                            state.write().await.ctx = Some(wrapped_ctx);
                            state.write().await.current_model = Some(message.clone());
                            Ok(status::Accepted(Some(Json(ModelEventResponse::OK {
                                message: None,
                            }))))
                        }
                        Err(_) => Err(status::Custom(
                            Status::InternalServerError,
                            Json(ModelEventResponse::ERROR {
                                message: Some("Unable to load model".to_owned()),
                            }),
                        )),
                    },
                    Err(_) => {
                        log!(Level::Error, "Unable to load model: Thread panicked");
                        Err(status::Custom(
                            Status::InternalServerError,
                            Json(ModelEventResponse::ERROR {
                                message: Some("Unable to load model".to_owned()),
                            }),
                        ))
                    }
                }
            } else {
                Err(status::Custom(
                    Status::BadRequest,
                    Json(ModelEventResponse::ERROR {
                        message: Some("Invalid model name".to_owned()),
                    }),
                ))
            }
        }
        ModelEventRequest::UNLOAD => {
            if let Some(wrapped_ctx) = state.read().await.ctx.clone() {
                let (sender, recv) = oneshot::channel::<bool>();
                state.write().await.ctx = None;
                rocket::tokio::spawn(async move {
                    let ctx = wrapped_ctx.lock().await.0;
                    drop(wrapped_ctx);
                    free_memory(ctx);
                    sender.send(true).ok();
                });
                match recv.await {
                    Ok(v) => {
                        if v {
                            state.write().await.current_model = None;
                            Ok(status::Accepted(Some(Json(ModelEventResponse::OK {
                                message: None,
                            }))))
                        } else {
                            Err(status::Custom(
                                Status::InternalServerError,
                                Json(ModelEventResponse::ERROR {
                                    message: Some(
                                        "Unable to unload: Error freeing memory".to_owned(),
                                    ),
                                }),
                            ))
                        }
                    }
                    Err(_) => Err(status::Custom(
                        Status::InternalServerError,
                        Json(ModelEventResponse::ERROR {
                            message: Some("Unable to unload: Thread panicked".to_owned()),
                        }),
                    )),
                }
            } else {
                Err(status::Custom(
                    Status::BadRequest,
                    Json(ModelEventResponse::ERROR { message: None }),
                ))
            }
        }
        ModelEventRequest::LIST => {
            let model_names =
                read_model_dir(&state.read().await.load_params.path_to_model_dir).await;
            if model_names.len() > 0 {
                Ok(status::Accepted(Some(Json(ModelEventResponse::OKVec {
                    message: model_names,
                }))))
            } else {
                Err(status::Custom(
                    Status::InternalServerError,
                    Json(ModelEventResponse::ERROR {
                        message: Some("No models in directory".to_owned()),
                    }),
                ))
            }
        }
        ModelEventRequest::CURRENT => Ok(status::Accepted(Some(Json(ModelEventResponse::OK {
            message: state.read().await.current_model.clone(),
        })))),
    }
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct CLI {
    // Path to model directory
    model_dir: PathBuf,
    #[arg(long)]
    // Context size [not designed to work for values greater than 2048] (default 2048)
    ctx_size: Option<i32>,
    #[arg(long)]
    // Number of layers to offload to GPU (default 0)
    gpu_offload_layers: Option<i32>,
    #[arg(long)]
    // Random seed (using GPU does not guarantee reproducible results) (default -1)
    seed: Option<i32>,
    #[arg(long)]
    // Use 16-bit floats for KV store (default true)
    use_f16: Option<bool>,
    #[arg(long)]
    // Pin model in memory for faster access (default true)
    use_mmap: Option<bool>,
    #[arg(long)]
    // Prevent mapped memory from going to disk (default false) (will cause errors if memory is insufficient)
    use_mlock: Option<bool>,
}

async fn read_model_dir(model_dir: &PathBuf) -> Vec<String> {
    let mut res: Vec<String> = Vec::new();
    match read_dir(model_dir).await {
        Ok(mut files) => {
            loop {
                match files.next_entry().await {
                    Ok(optional_file) => {
                        if let Some(file) = optional_file {
                            if file.file_name().into_string().is_ok()
                                && file.file_name().into_string().unwrap().ends_with(".bin")
                            {
                                res.push(file.file_name().into_string().unwrap());
                            }
                        } else {
                            break;
                        }
                    }
                    Err(io_error) => {
                        format!("Error reading file in model directory: {:?}", io_error);
                        continue;
                    }
                }
            }
            res
        }
        Err(_) => res,
    }
}

#[rocket::main]
async fn main() {
    let cli = CLI::parse();
    let load_params = LoadParams {
        context_size: cli.ctx_size,
        gpu_offload: cli.gpu_offload_layers,
        seed: cli.seed,
        kv_in_f16: cli.use_f16,
        no_swap: cli.use_mlock,
        pin_memory: cli.use_mmap,
        path_to_model_dir: cli.model_dir.clone(),
    };
    let models = read_model_dir(&load_params.path_to_model_dir).await;
    assert!(
        models.len() != 0,
        "No models found in {}",
        &load_params.path_to_model_dir.display()
    );
    println!("Initializing...");
    init();
    let res = rocket::build()
        .mount("/api/v1/", rocket::routes![change_model])
        .manage(RwLock::new(MainState {
            process_state: ProcessState::OK,
            has_output: false,
            current_token: String::new(),
            input_text: String::new(),
            load_params: load_params,
            ctx: None,
            current_model: None,
        }))
        .launch()
        .await;
}
