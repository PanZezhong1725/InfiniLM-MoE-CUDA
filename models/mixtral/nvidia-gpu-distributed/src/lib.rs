//#![cfg(detected_nccl)]

mod parameters;

#[macro_use]
extern crate log;

use causal_lm::{CausalLM, DecodingMeta, Model, QueryContext, SampleMeta};
use common::{f16, safe_tensors::SafeTensors, upos, utok, Blob, FileLoadError};
use common_nv::{
    cuda::{
        memcpy_d2h, AsRaw, Context, ContextResource, ContextSpore, DevByte, DevMem, DevMemSpore,
        Device, Stream, StreamSpore,
    },
    sample_nv, slice, split, udim, KernelsA, KernelsB, LocalSplitable, NvidiaKernels, Tensor,
};
use digit_layout::types::{F16, U32};
use itertools::izip;
use mixtral::{ConfigJson, MixtralConfig, MixtralParams};
use nccl::CommunicatorGroup;
use parameters::MixtralDistParams;
use tensor::{reslice, reslice_mut};

use std::{
    iter::{repeat, zip},
    mem::{take, ManuallyDrop},
    path::Path,
    slice::from_raw_parts,
    sync::Arc,
    time::Instant,
};

pub use common_nv::cuda;

pub struct MixtralGPU {
    config: MixtralConfig,

    comms: CommunicatorGroup,
    streams: Vec<StreamSpore>,
    kernels: NvidiaKernels,
    params: MixtralDistParams,
}

impl Model for MixtralGPU {
    type Meta = Vec<Device>;
    type Error = FileLoadError;

    #[inline]
    fn load(model_dir: impl AsRef<Path>, meta: Self::Meta) -> Result<Self, Self::Error> {
        let time = Instant::now();
        let config_json = ConfigJson::load(&model_dir)?;
        let config = MixtralConfig::new(&config_json);
        let host_params = MixtralParams::new(&config_json, SafeTensors::load_from_dir(model_dir)?);
        info!("load host: {:?}", time.elapsed());

        let kernels = NvidiaKernels::new(&meta, config.d as _);
        let contexts = meta
            .iter()
            .map(|dev| {
                dev.set_mempool_threshold(u64::MAX);
                dev.retain_primary()
            })
            .collect::<Vec<_>>();
        let comms = CommunicatorGroup::new(
            &meta
                .iter()
                .map(|dev| unsafe { dev.as_raw() })
                .collect::<Vec<_>>(),
        );
        let params = MixtralDistParams::load(&host_params, &config, &contexts);

        let streams = contexts
            .iter()
            .map(|context| context.apply(|ctx| ctx.stream().sporulate()))
            .collect::<Vec<_>>();
        Ok(Self {
            comms,
            streams,
            kernels,
            params,
            config: config,
        })
    }
}

impl CausalLM for MixtralGPU {
    type Storage = Cache;

    #[inline]
    fn max_seq_len(&self) -> upos {
        self.config.max_seq_len
    }
    #[inline]
    fn eos_token(&self) -> utok {
        self.config.eos_token
    }

    fn new_cache(&self) -> Tensor<Self::Storage> {
        let contexts = Arc::new(self.comms.contexts().collect::<Vec<_>>());
        let n = contexts.len() as udim;
        let dt = self.config.dtype;
        let shape = &[
            self.config.nlayers,
            2,
            self.config.nkvh / n,
            self.config.max_seq_len,
            self.config.dh,
        ];
        let bytes = shape.iter().product::<udim>() as usize * dt.nbytes();
        Tensor::new(
            dt,
            shape,
            Cache {
                mem: contexts
                    .iter()
                    .map(|context| context.apply(|ctx| ctx.malloc::<u8>(bytes).sporulate()))
                    .collect(),
                contexts,
            },
        )
    }

    fn duplicate_cache(&self, cache: &Tensor<Self::Storage>, pos: upos) -> Tensor<Self::Storage> {
        let contexts = Arc::new(self.comms.contexts().collect::<Vec<_>>());
        let n = contexts.len() as udim;
        let dt = self.config.dtype;
        let shape = &[
            self.config.nlayers,
            2,
            self.config.nkvh / n,
            self.config.max_seq_len,
            self.config.dh,
        ];
        let bytes = shape.iter().product::<udim>() as usize * dt.nbytes();
        assert!(pos <= self.config.max_seq_len);
        let mut new_cache = Tensor::new(
            dt,
            shape,
            Cache {
                mem: contexts
                    .iter()
                    .map(|context| context.apply(|ctx| ctx.malloc::<u8>(bytes).sporulate()))
                    .collect(),
                contexts: contexts.clone(),
            },
        );
        if pos > 0 {
            let slice = [
                slice![=>],
                slice![=>],
                slice![=>],
                slice![=>pos],
                slice![=>],
            ];
            for (i, context) in contexts.iter().enumerate() {
                context.apply(|ctx| {
                    let mut dst = Tensor::alloc(dt, shape, |_| {
                        &mut **new_cache.physical_mut().mem[i].sprout_mut(ctx)
                    })
                    .slice(&slice);
                    let src =
                        Tensor::alloc(dt, shape, |_| &**cache.physical().mem[i].sprout_ref(ctx))
                            .slice(&slice);
                    self.kernels.reform(&mut dst, &src, &ctx.stream());
                })
            }
        }

        new_cache
    }

    fn token_embed(&self, queries: impl IntoIterator<Item = utok>) -> Tensor<Self::Storage> {
        let tokens = queries.into_iter().collect::<Vec<_>>();
        let nt = tokens.len() as udim;

        let contexts = Arc::new(self.comms.contexts().collect::<Vec<_>>());
        let dt = self.config.dtype;
        let d = self.config.d;

        let mut x = Tensor::alloc(dt, &[nt, d], |len| malloc_all(&contexts, len));
        contexts[0].apply(|ctx| {
            let mut x = x.as_mut().map_physical(|u| &mut **u[0].sprout_mut(ctx));
            self.kernels.gather(
                &mut x,
                &self.params.embed_tokens(ctx),
                tokens,
                self.streams[0].sprout_ref(ctx),
            );
        });
        for (i, comm) in self.comms.call().iter().enumerate() {
            contexts[i].apply(|ctx| {
                let stream = self.streams[i].sprout_ref(ctx);
                let dst = x.physical_mut()[i].sprout_mut(ctx);
                comm.broadcast(dst, None, 0, stream);
            });
        }
        x.map_physical(|mem| Cache { contexts, mem })
    }

    fn forward<'a>(
        &self,
        queries: impl IntoIterator<Item = QueryContext<'a, Self::Storage>>,
        mut token_embedded: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage>
    where
        Self: 'a,
    {
        let queries = queries.into_iter().collect::<Vec<_>>();
        let mut nt = 0;
        let mut max_seq_len = 0;
        let mut max_att_len = 0;
        let seq_len = queries
            .iter()
            .map(|q| {
                let seq = q.seq_len();
                let att = q.att_len();
                nt += seq;
                max_seq_len = max_seq_len.max(seq);
                max_att_len = max_att_len.max(att);
                seq
            })
            .collect::<Vec<_>>();
        let seq_len = &seq_len;

        let dt = self.config.dtype;
        let d = self.config.d;
        let nh = self.config.nh;
        let nkvh = self.config.nkvh;
        let dh = d / nh;
        let dkv = nkvh * dh;
        let di = self.config.di;

        let n = self.comms.len() as udim;
        let reusing = (d + dkv + dkv).max(di + di);
        let pos = causal_lm::pos(&queries, nt);
        let pos = &pos;

        let x = token_embedded
            .as_mut()
            .map_physical(|u| unsafe { u.split() });
        let queries = queries
            .into_iter()
            .map(|q| {
                (
                    q.cache.map(|t| {
                        let ptrs = unsafe { t.physical_mut().split() };
                        Tensor::new(t.data_layout(), t.shape(), ptrs)
                    }),
                    q.range,
                )
            })
            .collect::<Vec<_>>();
        let queries = &queries;

        std::thread::scope(|s| {
            let _ = self
                .comms
                .iter()
                .enumerate()
                .map(|(i, comm)| {
                    let mut x = x.as_ref().map_physical(|u| unsafe {
                        std::slice::from_raw_parts_mut(u[i].0 as *mut DevByte, u[i].1)
                    });
                    let pos = pos.as_ref().map_physical(|u| &**u);
                    let mut queries = queries
                        .iter()
                        .map(|(cache, range)| {
                            (
                                cache.as_ref().map(|t| {
                                    t.as_ref().map_physical(|u| unsafe {
                                        std::slice::from_raw_parts_mut(
                                            u[i].0 as *mut DevByte,
                                            u[i].1,
                                        )
                                    })
                                }),
                                range,
                            )
                        })
                        .collect::<Vec<_>>();

                    s.spawn(move || {
                        comm.device().retain_primary().apply(|ctx| {
                            let mut queries = queries
                                .iter_mut()
                                .map(|(cache, range)| QueryContext {
                                    cache: cache.as_mut(),
                                    range: range.clone(),
                                })
                                .collect::<Vec<_>>();

                            let stream = self.streams[i].sprout_ref(ctx);

                            let pos = pos.map_physical(|u| stream.from_host(u));
                            let mut state_buf = Tensor::alloc(dt, &[nt, d + reusing / n], |len| {
                                stream.malloc::<u8>(len)
                            });
                            let buf_len_common = (nh / n * max_seq_len) as usize * dt.nbytes();
                            let mut q_buf = stream.malloc::<u8>(buf_len_common * dh as usize);
                            let mut att_buf =
                                stream.malloc::<u8>(buf_len_common * max_att_len as usize);
                            let mut routes_buf = Tensor::alloc(dt, &[nt, self.config.ne], |len| {
                                stream.malloc::<u8>(len)
                            });

                            for layer in 0..self.config.nlayers as usize {
                                self.self_att(
                                    &mut queries,
                                    seq_len,
                                    &mut x,
                                    &mut state_buf,
                                    &pos,
                                    &mut q_buf,
                                    &mut att_buf,
                                    i,
                                    layer,
                                    nt,
                                    stream,
                                );
                                comm.all_reduce(
                                    x.physical_mut(),
                                    None,
                                    self.config.dtype,
                                    nccl::ReduceType::ncclSum,
                                    stream,
                                );

                                self.moe(&mut x, &mut state_buf, &mut routes_buf, i, layer, stream);
                                comm.all_reduce(
                                    x.physical_mut(),
                                    None,
                                    self.config.dtype,
                                    nccl::ReduceType::ncclSum,
                                    stream,
                                );
                            }

                            pos.take_physical().drop_on(stream);
                            att_buf.drop_on(stream);
                            q_buf.drop_on(stream);
                            state_buf.take_physical().drop_on(stream);
                            routes_buf.take_physical().drop_on(stream);
                        })
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|t| t.join().unwrap())
                .collect::<Vec<_>>();
        });
        token_embedded
    }

    fn decode(
        &self,
        decoding: impl IntoIterator<Item = DecodingMeta>,
        mut hidden_state: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage> {
        let dt = self.config.dtype;
        let d = self.config.d;

        let contexts = Arc::new(vec![self.comms.contexts().next().unwrap()]);
        let ans = contexts[0].apply(|ctx| {
            let stream = self.streams[0].sprout_ref(ctx);

            let mut x = hidden_state
                .as_mut()
                .map_physical(|u| &mut **u.mem[0].sprout_mut(ctx));
            let range = DecodingMeta::select(&mut x, decoding, |dst, src| {
                stream.memcpy_d2d(dst, src);
            });

            if range.is_empty() {
                return Tensor::alloc(dt, &[0, d as _], |_| Cache {
                    contexts: contexts.clone(),
                    mem: vec![stream.malloc::<u8>(0).sporulate()],
                });
            }

            let model_norm = self.params.lm_layernorm(ctx);
            let lm_head = self.params.lm_head(ctx).transpose(&[1, 0]);

            let mut x = x.slice(&[slice![range.start => range.end], slice![=>]]);
            let mut logits = Tensor::alloc(dt, &[x.shape()[0], lm_head.shape()[1]], |len| {
                stream.malloc::<u8>(len)
            });

            // 复制一个 x 以实现原地归一化
            let x_ = x
                .as_ref()
                .map_physical(|u| unsafe { from_raw_parts(u.as_ptr(), u.len()) });
            self.kernels
                .rms_norm(&mut x, &x_, &model_norm, self.config.epsilon, stream);
            self.kernels
                .mat_mul(&mut logits, 0., &x, &lm_head, 1., stream);

            logits.map_physical(|u| Cache {
                contexts: contexts.clone(),
                mem: vec![u.sporulate()],
            })
        });

        take(&mut ManuallyDrop::new(hidden_state.take_physical()).mem)
            .into_iter()
            .zip(self.comms.contexts())
            .enumerate()
            .for_each(|(i, (mem, context))| {
                context.apply(|ctx| mem.sprout(ctx).drop_on(self.streams[i].sprout_ref(ctx)));
            });

        ans
    }

    fn sample(
        &self,
        args: impl IntoIterator<Item = SampleMeta>,
        logits: Tensor<Self::Storage>,
    ) -> Vec<utok> {
        assert_eq!(logits.data_layout(), F16);
        let &[_nt, voc] = logits.shape() else {
            panic!()
        };
        let voc = voc as usize;
        let Cache { contexts, mem } = logits.physical();

        contexts[0].apply(|ctx| {
            sample_nv(
                args.into_iter()
                    .flat_map(|meta| repeat(meta.args).take(meta.num_decode))
                    .enumerate(),
                mem[0].sprout_ref(ctx),
                voc,
                self.streams[0].sprout_ref(ctx),
            )
        })
    }
}

impl MixtralGPU {
    fn self_att(
        &self,
        queries: &mut [QueryContext<&mut [DevByte]>],
        seq_len: &[udim],
        x: &mut Tensor<&mut [DevByte]>,
        state_buf: &mut Tensor<DevMem>,
        pos: &Tensor<DevMem>,
        q_buf: &mut DevMem,
        att_buf: &mut DevMem,
        i: usize,
        layer: usize,
        nt: udim,
        stream: &Stream,
    ) {
        let dt = self.config.dtype;
        let d = self.config.d;
        let nh = self.config.nh;
        let nkvh = self.config.nkvh;
        let dh = d / nh;
        let dkv = nkvh * dh;
        let di = self.config.di;
        let head_group = nh / nkvh;
        let head_div = (dh as f32).sqrt().recip();
        let theta = self.config.theta;
        let epsilon = self.config.epsilon;

        let n = self.comms.len() as udim;
        let reusing = (d + dkv + dkv).max(di + di);

        let ctx = stream.ctx();
        let kernels = &self.kernels;

        let (mut x1, qkv) = split!(state_buf.as_mut().map_physical(|u| LocalSplitable::from(&mut **u)); [1]: d, reusing / n);
        let mut qkv = qkv.slice(&[slice![=>], slice![=> (d + dkv + dkv) / n]]);
        let qkv_proj = self.params.qkv_proj(layer, i, ctx);
        kernels.rms_norm(
            &mut x1,
            x,
            &self.params.att_norm(layer, i, ctx),
            epsilon,
            stream,
        );

        kernels.mat_mul(&mut qkv, 0., &x1, &qkv_proj.transpose(&[1, 0]), 1., stream);

        let (q, k, v) = split!(qkv; [1]: d / n, dkv / n, dkv / n);
        let mut q = q.reshape(&[nt, nh / n, dh]);
        let mut k = k.reshape(&[nt, nkvh / n, dh]);
        let v = v.reshape(&[nt, nkvh / n, dh]);
        let o = x1.reshape(&[nt, nh, dh]);
        let o = o.slice(&[slice![=>], slice![=> nh / n], slice![=>]]);

        kernels.rope(&mut q, pos, theta, stream);
        kernels.rope(&mut k, pos, theta, stream);

        let q = q.transpose(&[1, 0, 2]).split(1, seq_len);
        let k = k.transpose(&[1, 0, 2]).split(1, seq_len);
        let v = v.transpose(&[1, 0, 2]).split(1, seq_len);
        let o = o.transpose(&[1, 0, 2]).split(1, seq_len);

        for (query, q, k, v, mut o) in izip!(queries, q, k, v, o) {
            let pos = query.pos();
            let seq_len = query.seq_len();
            let att_len = query.att_len();
            let Some((mut k_cache, mut v_cache)) = query.cache(layer as _) else {
                continue;
            };

            let slice_cat = &[slice![=>], slice![pos =>=> seq_len], slice![=>]];
            let slice_att = &[slice![=>], slice![      => att_len], slice![=>]];
            let shape_q0 = &[nkvh / n * head_group, seq_len, dh];
            let shape_q1 = &[nkvh / n, head_group * seq_len, dh];
            let shape_att0 = &[nkvh / n, head_group * seq_len, att_len];
            let shape_att1 = &[nkvh / n * head_group, seq_len, att_len];

            let mut q_att = Tensor::new(dt, shape_q0, &mut q_buf[..]);
            let mut k_cat = k_cache.as_mut().slice(slice_cat).map_physical(|u| &mut **u);
            let mut v_cat = v_cache.as_mut().slice(slice_cat).map_physical(|u| &mut **u);
            kernels.reform(&mut q_att, &q, stream);
            kernels.reform(&mut k_cat, &k, stream);
            kernels.reform(&mut v_cat, &v, stream);

            let q_att = q_att.reshape(shape_q1);
            let k_att = k_cache.slice(slice_att).transpose(&[0, 2, 1]);
            let v_att = v_cache.slice(slice_att);

            let mut att = Tensor::new(dt, shape_att0, &mut att_buf[..]);
            kernels.mat_mul(&mut att, 0., &q_att, &k_att, head_div, stream);
            let mut att = att.reshape(shape_att1);
            kernels.softmax(&mut att, stream);
            let att = att.reshape(shape_att0);
            let mut x2 = q_att;
            kernels.mat_mul(&mut x2, 0., &att, &v_att, 1., stream);

            kernels.reform(&mut o, &x2.reshape(shape_q0), stream);
        }

        let (x1, _) = split!(state_buf.as_mut().map_physical(|u| LocalSplitable::from(&mut **u)); [1]: d, reusing / n);

        let o = x1.as_ref().slice(&[slice![=>], slice![=> d/n as udim]]);
        let o = o.map_physical(|u| &**u);
        let o_proj = self.params.o_proj(layer, i, ctx).transpose(&[1, 0]);
        kernels.mat_mul(x, if i == 0 { 1. } else { 0. }, &o, &o_proj, 1., stream);
    }

    fn moe(
        &self,
        x: &mut Tensor<&mut [DevByte]>,
        state_buf: &mut Tensor<DevMem>,
        routes_buf: &mut Tensor<DevMem>,
        i: usize,
        layer: usize,
        stream: &Stream,
    ) {
        let d = self.config.d;
        let nh = self.config.nh;
        let nkvh = self.config.nkvh;
        let dh = d / nh;
        let dkv = nkvh * dh;
        let di = self.config.di;
        let epsilon = self.config.epsilon;
        let ctx = stream.ctx();
        let nt = x.shape()[0];
        let kernels = &self.kernels;
        let n = self.comms.len() as udim;
        let reusing = (d + dkv + dkv).max(di + di);

        // x residual
        // x1 post layernorm
        let (mut x1, gate_up) = split!(state_buf.as_mut().map_physical(|u| LocalSplitable::from(&mut **u)); [1]: d, reusing / n);
        let gate_up = gate_up.slice(&[slice![=>], slice![=> (di + di) / n]]);

        kernels.rms_norm(
            &mut x1,
            x,
            &self.params.moe().layernorm(layer, i, ctx),
            epsilon,
            stream,
        );

        let w_moe_gate = self.params.moe().gate(layer, i, ctx).transpose(&[1, 0]);
        kernels.mat_mul(routes_buf, 0., &x1, &w_moe_gate, 1., stream);

        let mut routes = routes_buf.as_ref().map_physical(|s| {
            let mut host = Blob::new(s.len());
            memcpy_d2h(&mut host, s);
            host
        });
        softmax(&mut routes);
        let mut moe_w = Tensor::alloc(routes.data_layout(), &[nt, self.config.k], Blob::new);
        let mut moe_i = Tensor::alloc(U32, &[nt, self.config.k], Blob::new);
        topk(&routes, self.config.k as _, &mut moe_w, &mut moe_i);
        let weights: &[f16] = reslice(moe_w.as_slice());
        let indices: &[u32] = reslice(moe_i.as_slice());

        let shard = vec![1; x.shape()[0] as _];
        let x = x.as_mut().map_physical(|u| LocalSplitable::from(&mut **u));
        let mut _x0 = x.split(0, &shard);
        let mut _x1 = x1.split(0, &shard);
        let mut _gate_up = gate_up.split(0, &shard);
        for tok in (0..nt).rev() {
            let sum: f32 = (0..self.config.k)
                .map(|k| weights[(tok * self.config.k + k) as usize].to_f32())
                .sum();
            let mut gate_up_slice = _gate_up.pop_back().unwrap();
            let mut x0_slice = _x0.pop_back().unwrap();
            let x1_slice = _x1.pop_back().unwrap();
            let mut residual = true;
            for k in 0..self.config.k {
                let expert = indices[(tok * self.config.k + k) as usize] as usize;
                let expert_w = weights[(tok * self.config.k + k) as usize].to_f32() / sum;
                let w_gate_up = self
                    .params
                    .moe()
                    .mlp_gate_up(layer, expert, i, ctx)
                    .transpose(&[1, 0]);
                let w_down = self
                    .params
                    .moe()
                    .mlp_down(layer, expert, i, ctx)
                    .transpose(&[1, 0]);

                kernels.mat_mul(&mut gate_up_slice, 0., &x1_slice, &w_gate_up, 1., stream);
                let (mut gate, up) = split!(gate_up_slice; [1]: di / n, di / n);
                kernels.swiglu(&mut gate, &up, stream);
                kernels.mat_mul(
                    &mut x0_slice,
                    if i != 0 && residual { 0. } else { 1. },
                    &gate,
                    &w_down,
                    expert_w,
                    stream,
                );
                residual = false;
            }
        }
    }
}

impl Drop for MixtralGPU {
    #[inline]
    fn drop(&mut self) {
        let contexts = self.comms.contexts().collect::<Vec<_>>();

        self.params.kill(&contexts);
        for (context, stream) in zip(contexts, std::mem::take(&mut self.streams)) {
            context.apply(|ctx| drop(stream.sprout(ctx)));
        }
    }
}

pub struct Cache {
    pub contexts: Arc<Vec<Context>>,
    pub mem: Vec<DevMemSpore>,
}

impl Cache {
    unsafe fn split(&mut self) -> Vec<(cuda::bindings::CUdeviceptr, usize)> {
        self.mem
            .iter()
            .map(|mem| (mem.as_raw(), mem.len()))
            .collect()
    }
}

impl Drop for Cache {
    #[inline]
    fn drop(&mut self) {
        let mem = std::mem::take(&mut self.mem);
        for (context, mem) in zip(&*self.contexts, mem) {
            context.apply(|ctx| drop(mem.sprout(ctx)));
        }
    }
}

fn malloc_all(contexts: &[Context], len: usize) -> Vec<DevMemSpore> {
    contexts
        .iter()
        .map(|context| context.apply(|ctx| ctx.malloc::<u8>(len).sporulate()))
        .collect()
}

fn topk(logits: &Tensor<Blob>, k: usize, weight: &mut Tensor<Blob>, indices: &mut Tensor<Blob>) {
    let n = logits.shape()[0];
    let dim = logits.shape()[1];
    let slice = logits.as_slice();
    let slice: &[f16] = reslice(slice);
    let weight_slice: &mut [f16] = reslice_mut(weight.physical_mut());
    let indices_slice: &mut [u32] = reslice_mut(indices.physical_mut());
    for token_i in 0..n {
        #[derive(PartialEq, Debug)]
        struct WithIndex {
            idx: usize,
            data: f16,
        }
        impl PartialOrd for WithIndex {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Eq for WithIndex {}
        impl Ord for WithIndex {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.data.total_cmp(&other.data).reverse()
            }
        }

        let line = &slice[(token_i * dim) as usize..][..dim as usize];
        // let mut heap = BinaryHeap::<WithIndex>::new();
        let mut vec = line
            .iter()
            .enumerate()
            .map(|(idx, &data)| WithIndex { idx, data })
            .collect::<Vec<_>>();
        vec.sort_unstable();
        let top = &vec[..k];
        for top_i in 0..k {
            weight_slice[(token_i as usize) * k + top_i] = top[top_i].data;
            indices_slice[(token_i as usize) * k + top_i] = top[top_i].idx as u32;
        }
    }
}

fn softmax(logits: &mut Tensor<Blob>) {
    let n = logits.shape()[0];
    let dim = logits.shape()[1];
    let slice: &mut [f16] = reslice_mut(logits.physical_mut());
    for token_i in 0..n {
        let line: &mut [f16] = &mut slice[(token_i * dim) as usize..][..dim as usize];
        let max = line.iter().cloned().fold(f16::NEG_INFINITY, f16::max);
        let mut exp_sum = 0.;
        for i in 0..dim {
            let val = (line[i as usize] - max).to_f32().exp();
            exp_sum += val;
        }
        for i in 0..dim {
            let val = (line[i as usize] - max).to_f32().exp();
            line[i as usize] = f16::from_f32(val / exp_sum);
        }
    }
}

fn debug<T>(tensor: &Tensor<T>)
where
    T: std::ops::Deref<Target = [cuda::DevByte]>,
{
    println!(
        "{}",
        tensor.as_ref().map_physical(|s| {
            let mut host = Blob::new(s.len());
            memcpy_d2h(&mut host, s);
            host
        })
    );
}
