mod infer;

use causal_lm::Model;
use common::{safe_tensors::SafeTensors, FileLoadError};
use common_cpu::CpuKernels;
use mixtral::{ConfigJson, MixtralConfig, MixtralParams};
use std::path::Path;

pub struct MixtralCPU {
    config: MixtralConfig,
    params: MixtralParams,

    kernels: CpuKernels,
}

impl Model for MixtralCPU {
    type Error = FileLoadError;
    type Meta = ();

    fn load(model_dir: impl AsRef<Path>, _: Self::Meta) -> Result<Self, Self::Error> {
        let config_json = ConfigJson::load(&model_dir)?;
        let config = MixtralConfig::new(&config_json);
        Ok(Self {
            config,
            params: MixtralParams::new(&config_json, SafeTensors::load_from_dir(model_dir)?),
            kernels: Default::default(),
        })
    }
}

#[test]
fn test_build() {
    use std::time::Instant;

    let t0 = Instant::now();
    let _transformer = MixtralCPU::load(
        "/data1/shared/hugging_face/Mixtral-8x7B-Instruct-v0.1_F16/",
        (),
    );
    let t1 = Instant::now();
    println!("build transformer {:?}", t1 - t0);
}
