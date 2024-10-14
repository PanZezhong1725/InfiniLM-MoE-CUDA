use common::Blob;
use common_nv::cuda::{ContextSpore, CurrentCtx, DevByte};
use common_nv::slice;
use common_nv::{
    cuda::{Context, ContextResource, DevMemSpore, HostMemSpore},
    udim, Tensor,
};
use mixtral::{MixtralConfig, MixtralParams};
use std::mem::ManuallyDrop;
pub struct MixtralDistParams(ManuallyDrop<_MixtralDistParams>);
struct _MixtralDistParams {
    embed_tokens: Tensor<HostMemSpore>,
    decoders: Vec<DecoderLayer>,
    moe: MoEParams,
    lm_layernorm: Tensor<DevMemSpore>,
    lm_head: Tensor<DevMemSpore>,
}

struct DecoderLayer {
    layernorm: Vec<Tensor<DevMemSpore>>,
    w_qkv: Vec<Tensor<DevMemSpore>>,
    w_o: Vec<Tensor<DevMemSpore>>,
}

pub struct MoEParams {
    moe_layers: Vec<MoE>,
}

struct MoE {
    layernorm: Vec<Tensor<DevMemSpore>>,
    gate: Vec<Tensor<DevMemSpore>>,
    experts: Vec<MLP>,
}

struct MLP {
    gate_up: Vec<Tensor<DevMemSpore>>,
    down: Vec<Tensor<DevMemSpore>>,
}

impl _MixtralDistParams {
    pub fn load(host_params: &MixtralParams, config: &MixtralConfig, contexts: &[Context]) -> Self {
        let dt = config.dtype;
        let d = config.d;
        let di = config.di;
        let dk = config.dh * config.nkvh;
        let n_devices = contexts.len();
        let slice_dh = d / n_devices as udim;
        let slice_dkv = dk / n_devices as udim;
        let slice_di = di / n_devices as udim;
        let mut decoders = Vec::with_capacity(config.nlayers as _);
        let mut moe_layers = Vec::with_capacity(config.nlayers as _);

        // Copy embed_tokens, lm_layernorm, and lm_head to device 0
        let (embed_tokens, lm_layernorm, lm_head) = contexts[0].apply(|ctx| {
            (
                host_params.embed_tokens().map_physical(|m| {
                    let mut host = ctx.malloc_host::<u8>(m.len());
                    host.clone_from_slice(&m);
                    host.sporulate().into()
                }),
                host_params
                    .model_norm()
                    .map_physical(|m| ctx.from_host(&m).sporulate().into()),
                host_params
                    .lm_head()
                    .map_physical(|m| ctx.from_host(&m).sporulate().into()),
            )
        });

        for layer in 0..config.nlayers {
            info!("loading layer {layer}/{}", config.nlayers - 1);
            let mut att_layernorm = Vec::with_capacity(n_devices);
            let mut att_qkv = Vec::with_capacity(n_devices);
            let mut att_o = Vec::with_capacity(n_devices);
            let mut moe_layernorm = Vec::with_capacity(n_devices);
            let mut moe_gate = Vec::with_capacity(n_devices);
            let mut mlps = Vec::with_capacity(config.ne as _);
            for _ in 0..config.ne {
                let gate_up = Vec::with_capacity(n_devices);
                let down = Vec::with_capacity(n_devices);
                mlps.push(MLP { gate_up, down });
            }
            for device in 0..n_devices {
                contexts[device].apply(|ctx| {
                    // attention layernorm, replicated
                    att_layernorm.push(
                        host_params
                            .input_layernorm(layer)
                            .map_physical(|m| ctx.from_host(&m).sporulate().into()),
                    );
                    // qkv_proj, sharded along kv heads
                    let mut qkv_proj_slice =
                        Blob::new((d * (slice_dh + slice_dkv * 2)) as usize * dt.nbytes());
                    host_params
                        .w_qkv(layer)
                        .slice(&[slice![slice_dh * device as udim =>=> slice_dh], slice![=>]])
                        .reform_to(&mut Tensor::new(
                            dt,
                            &[slice_dh, d],
                            &mut qkv_proj_slice[..][..(slice_dh * d) as usize * dt.nbytes()],
                        ));
                    host_params
                        .w_qkv(layer)
                        .slice(&[
                            slice![d + slice_dkv * device as udim =>=> slice_dkv],
                            slice![=>],
                        ])
                        .reform_to(&mut Tensor::new(
                            dt,
                            &[slice_dkv, d],
                            &mut qkv_proj_slice[(d * slice_dh) as usize * dt.nbytes()..]
                                [..(slice_dkv * d) as usize * dt.nbytes()],
                        ));
                    host_params
                        .w_qkv(layer)
                        .slice(&[
                            slice![d + dk + slice_dkv * device as udim =>=> slice_dkv],
                            slice![=>],
                        ])
                        .reform_to(&mut Tensor::new(
                            dt,
                            &[slice_dkv, d],
                            &mut qkv_proj_slice
                                [(d * (slice_dh + slice_dkv)) as usize * dt.nbytes()..]
                                [..(slice_dkv * d) as usize * dt.nbytes()],
                        ));
                    att_qkv.push(
                        Tensor::new(dt, &[slice_dh + slice_dkv * 2, d], qkv_proj_slice)
                            .map_physical(|m| ctx.from_host(&m).sporulate().into()),
                    );
                    // o_proj, sharded along attention heads
                    let mut o_proj_slice = Tensor::alloc(dt, &[d, slice_dh], Blob::new);
                    host_params
                        .w_o(layer)
                        .slice(&[slice![=>], slice![slice_dh * device as udim =>=> slice_dh]])
                        .reform_to(&mut o_proj_slice);
                    att_o.push(o_proj_slice.map_physical(|m| ctx.from_host(&m).sporulate().into()));
                    // final layernorm, replicated
                    moe_layernorm.push(
                        host_params
                            .post_attention_layernorm(layer)
                            .map_physical(|m| ctx.from_host(&m).sporulate().into()),
                    );
                    // moe gate, replicated
                    moe_gate.push(
                        host_params
                            .moe_gate(layer)
                            .map_physical(|m| ctx.from_host(&m).sporulate().into()),
                    );

                    for expert in 0..config.ne as usize {
                        // mlp_gate_up, sharded along intermediate dimension
                        let mut gate_up_slice =
                            Blob::new((d * slice_di * 2) as usize * dt.nbytes());
                        unsafe {
                            host_params
                                .mlp_gate_up(layer, expert.try_into().unwrap())
                                .slice(&[
                                    slice![slice_di * device as udim =>=> slice_di],
                                    slice![=>],
                                ])
                                .reform_to_raw(
                                    &mut gate_up_slice[..][..(d * slice_di) as usize * dt.nbytes()],
                                )
                        };
                        unsafe {
                            host_params
                                .mlp_gate_up(layer, expert.try_into().unwrap())
                                .slice(&[
                                    slice![di + slice_di * device as udim =>=> slice_di],
                                    slice![=>],
                                ])
                                .reform_to_raw(
                                    &mut gate_up_slice[(d * slice_di) as usize * dt.nbytes()..]
                                        [..(d * slice_di) as usize * dt.nbytes()],
                                )
                        };
                        mlps.get_mut(expert).unwrap().gate_up.push(
                            Tensor::new(dt, &[slice_di * 2, d], gate_up_slice)
                                .map_physical(|m| ctx.from_host(&m).sporulate().into()),
                        );
                        // mlp_down, sharded along intermediate dimension
                        let mut down_slice = Tensor::alloc(dt, &[d, slice_di], Blob::new);
                        host_params
                            .mlp_down(layer, expert.try_into().unwrap())
                            .slice(&[slice![=>], slice![slice_di * device as udim=>=> slice_di]])
                            .reform_to(&mut down_slice);
                        mlps.get_mut(expert).unwrap().down.push(
                            down_slice.map_physical(|m| ctx.from_host(&m).sporulate().into()),
                        );
                    }
                });
            }

            decoders.push(DecoderLayer {
                layernorm: att_layernorm,
                w_qkv: att_qkv,
                w_o: att_o,
            });
            moe_layers.push(MoE {
                layernorm: moe_layernorm,
                gate: moe_gate,
                experts: mlps,
            });
        }

        _MixtralDistParams {
            embed_tokens: embed_tokens,
            decoders: decoders,
            moe: MoEParams { moe_layers },
            lm_layernorm: lm_layernorm,
            lm_head: lm_head,
        }
    }
}

impl MixtralDistParams {
    pub fn load(host_params: &MixtralParams, config: &MixtralConfig, contexts: &[Context]) -> Self {
        MixtralDistParams(ManuallyDrop::new(_MixtralDistParams::load(
            host_params,
            config,
            contexts,
        )))
    }

    pub fn embed_tokens<'ctx>(&'ctx self, ctx: &'ctx CurrentCtx) -> Tensor<&[u8]> {
        self.0
            .embed_tokens
            .as_ref()
            .map_physical(|m| &**m.sprout_ref(ctx))
    }

    pub fn lm_layernorm<'ctx>(&'ctx self, ctx: &'ctx CurrentCtx) -> Tensor<&[DevByte]> {
        self.0
            .lm_layernorm
            .as_ref()
            .map_physical(|m| &**m.sprout_ref(ctx))
    }

    pub fn lm_head<'ctx>(&'ctx self, ctx: &'ctx CurrentCtx) -> Tensor<&[DevByte]> {
        self.0
            .lm_head
            .as_ref()
            .map_physical(|m| &**m.sprout_ref(ctx))
    }

    pub fn att_norm<'ctx>(
        &'ctx self,
        layer: usize,
        device: usize,
        ctx: &'ctx CurrentCtx,
    ) -> Tensor<&[DevByte]> {
        self.0.decoders[layer].layernorm[device]
            .as_ref()
            .map_physical(|m| &**m.sprout_ref(ctx))
    }

    pub fn qkv_proj<'ctx>(
        &'ctx self,
        layer: usize,
        device: usize,
        ctx: &'ctx CurrentCtx,
    ) -> Tensor<&[DevByte]> {
        self.0.decoders[layer].w_qkv[device]
            .as_ref()
            .map_physical(|m| &**m.sprout_ref(ctx))
    }

    pub fn o_proj<'ctx>(
        &'ctx self,
        layer: usize,
        device: usize,
        ctx: &'ctx CurrentCtx,
    ) -> Tensor<&[DevByte]> {
        self.0.decoders[layer].w_o[device]
            .as_ref()
            .map_physical(|m| &**m.sprout_ref(ctx))
    }

    pub fn moe(&self) -> &MoEParams {
        &self.0.moe
    }

    pub fn kill(&mut self, contexts: &[Context]) {
        let mut params = unsafe { ManuallyDrop::take(&mut self.0) };
        contexts.iter().next().unwrap().apply(|ctx| {
            drop(params.embed_tokens.take_physical().sprout(ctx));
            drop(params.lm_layernorm.take_physical().sprout(ctx));
            drop(params.lm_head.take_physical().sprout(ctx));
        });

        for _ in 0..params.decoders.len() {
            let mut decoder = params.decoders.pop().unwrap();
            let mut moe = params.moe.moe_layers.pop().unwrap();
            for context in contexts.iter().rev() {
                context.apply(|ctx| {
                    drop(decoder.layernorm.pop().unwrap().take_physical().sprout(ctx));
                    drop(decoder.w_qkv.pop().unwrap().take_physical().sprout(ctx));
                    drop(decoder.w_o.pop().unwrap().take_physical().sprout(ctx));

                    drop(moe.layernorm.pop().unwrap().take_physical().sprout(ctx));
                    drop(moe.gate.pop().unwrap().take_physical().sprout(ctx));
                });
            }
            for _ in 0..moe.experts.len() {
                let mut expert = moe.experts.pop().unwrap();
                for context in contexts.iter().rev() {
                    context.apply(|ctx| {
                        drop(expert.gate_up.pop().unwrap().take_physical().sprout(ctx));
                        drop(expert.down.pop().unwrap().take_physical().sprout(ctx));
                    });
                }
            }
        }
    }
}

impl MoEParams {
    pub fn layernorm<'ctx>(
        &'ctx self,
        layer: usize,
        device: usize,
        ctx: &'ctx CurrentCtx,
    ) -> Tensor<&[DevByte]> {
        self.moe_layers[layer].layernorm[device]
            .as_ref()
            .map_physical(|m| &**m.sprout_ref(ctx))
    }

    pub fn gate<'ctx>(
        &'ctx self,
        layer: usize,
        device: usize,
        ctx: &'ctx CurrentCtx,
    ) -> Tensor<&[DevByte]> {
        self.moe_layers[layer].gate[device]
            .as_ref()
            .map_physical(|m| &**m.sprout_ref(ctx))
    }

    pub fn mlp_gate_up<'ctx>(
        &'ctx self,
        layer: usize,
        expert: usize,
        device: usize,
        ctx: &'ctx CurrentCtx,
    ) -> Tensor<&[DevByte]> {
        self.moe_layers[layer].experts[expert].gate_up[device]
            .as_ref()
            .map_physical(|m| &**m.sprout_ref(ctx))
    }

    pub fn mlp_down<'ctx>(
        &'ctx self,
        layer: usize,
        expert: usize,
        device: usize,
        ctx: &'ctx CurrentCtx,
    ) -> Tensor<&[DevByte]> {
        self.moe_layers[layer].experts[expert].down[device]
            .as_ref()
            .map_physical(|m| &**m.sprout_ref(ctx))
    }
}

#[test]
fn test_load() {
    use common::safe_tensors::SafeTensors;
    use common_nv::cuda::{self, Device};
    use mixtral::ConfigJson;
    use std::time::Instant;

    let n = 4;
    cuda::init();
    if Device::count() < n {
        return;
    }
    let contexts = (0..n as _)
        .map(|i| Device::new(i).retain_primary())
        .collect::<Vec<_>>();
    let model_dir = "/data1/shared/hugging_face/Mixtral-8x7B-Instruct-v0.1_F16/";
    let time = Instant::now();
    let config_json = ConfigJson::load(&model_dir).unwrap();
    let config = MixtralConfig::new(&config_json);
    let host_params = MixtralParams::new(
        &config_json,
        SafeTensors::load_from_dir(&model_dir).unwrap(),
    );
    let mut _params = MixtralDistParams::load(&host_params, &config, &contexts);
    println!("Time {:?}", time.elapsed());
    _params.kill(&contexts);
}
