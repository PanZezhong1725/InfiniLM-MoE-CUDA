use common::{utok, FileLoadError};
use digit_layout::{
    types::{BF16, F16, F32},
    DigitLayout,
};
use std::{fs, path::Path};
use tensor::udim;
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct ConfigJson {
    pub bos_token_id: utok,
    pub eos_token_id: utok,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub torch_dtype: String,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
}

impl ConfigJson {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, FileLoadError> {
        let path = model_dir.as_ref().join("config.json");
        let content = fs::read_to_string(path).map_err(FileLoadError::Io)?;
        serde_json::from_str(&content).map_err(FileLoadError::Json)
    }

    pub fn data_layout(&self) -> DigitLayout {
        match self.torch_dtype.as_str() {
            "float16" => F16,
            "float32" => F32,
            "bfloat16" => BF16,
            _ => todo!(),
        }
    }
}

#[inline(always)]
const fn default_rms_norm_eps() -> f32 {
    1e-5
}

#[inline(always)]
const fn default_rope_theta() -> f32 {
    1e4
}

#[derive(Clone, Debug)]
pub struct MixtralConfig {
    pub dtype: DigitLayout,
    pub voc: udim,
    pub nlayers: udim,
    pub nh: udim,
    pub nkvh: udim,
    pub d: udim,
    pub dh: udim,
    pub di: udim,
    pub ne: udim,
    pub k: udim,
    pub max_seq_len: udim,
    pub bos_token: utok,
    pub eos_token: utok,
    pub epsilon: f32,
    pub theta: f32,
}

impl MixtralConfig {
    pub fn new(config: &ConfigJson) -> Self {
        MixtralConfig {
            dtype: config.data_layout(),
            voc: config.vocab_size as udim,
            nlayers: config.num_hidden_layers as udim,
            nh: config.num_attention_heads as udim,
            nkvh: config.num_key_value_heads as udim,
            d: config.hidden_size as udim,
            dh: config.hidden_size as udim / config.num_attention_heads as udim,
            di: config.intermediate_size as udim,
            max_seq_len: config.max_position_embeddings as udim,
            bos_token: config.bos_token_id,
            eos_token: config.eos_token_id,
            epsilon: config.rms_norm_eps,
            theta: config.rope_theta,
            ne: config.num_local_experts as udim,
            k: config.num_experts_per_tok as udim,
        }
    }
}
