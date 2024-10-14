use super::MixtralCPU;
use causal_lm::{CausalLM, DecodingMeta, QueryContext, SampleMeta};
use common::{f16, upos, utok, Blob};
use common_cpu::{KernelsA, KernelsB, ThisThread};
use digit_layout::{types::U32, DigitLayout};
use itertools::izip;
use std::{iter::repeat, slice::from_raw_parts};
use tensor::{reslice, reslice_mut, slice, split, udim, LocalSplitable, Tensor};

impl CausalLM for MixtralCPU {
    type Storage = Blob;

    #[inline]
    fn eos_token(&self) -> utok {
        self.config.eos_token
    }
    #[inline]
    fn max_seq_len(&self) -> upos {
        self.config.max_seq_len
    }

    fn new_cache(&self) -> Tensor<Self::Storage> {
        let dt = self.config.dtype;
        let nlayers = self.config.nlayers;
        let nkvh = self.config.nkvh;
        let max_seq_len = self.config.max_seq_len;
        let d = self.config.d;
        let nh = self.config.nh;
        Tensor::alloc(dt, &[nlayers, 2, nkvh, max_seq_len, d / nh], Blob::new)
    }

    fn duplicate_cache(&self, cache: &Tensor<Self::Storage>, pos: upos) -> Tensor<Self::Storage> {
        let &[_nlayers, 2, _nkvh, max_seq_len, _dh] = cache.shape() else {
            panic!()
        };
        assert!(pos <= max_seq_len);
        let slice = [
            slice![=>],
            slice![=>],
            slice![=>],
            slice![=>pos],
            slice![=>],
        ];

        let mut ans = Tensor::alloc(cache.data_layout(), cache.shape(), Blob::new);
        cache
            .as_ref()
            .slice(&slice)
            .map_physical(|u| &**u)
            .reform_to(&mut ans.as_mut().slice(&slice).map_physical(|u| &mut **u));
        ans
    }

    fn token_embed(&self, queries: impl IntoIterator<Item = utok>) -> Tensor<Self::Storage> {
        let dt = self.config.dtype;
        let d = self.config.d;

        let tokens = queries.into_iter().collect::<Vec<_>>();
        let nt = tokens.len() as udim;

        let mut x = Tensor::alloc(dt, &[nt, d], Blob::new);
        self.kernels
            .gather(&mut x, &self.params.embed_tokens(), tokens, &ThisThread);
        x
    }

    fn forward<'a>(
        &self,
        queries: impl IntoIterator<Item = QueryContext<'a, Self::Storage>>,
        token_embedded: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage>
    where
        Self: 'a,
    {
        let mut queries = queries.into_iter().collect::<Vec<_>>();
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

        let dt = self.config.dtype;
        let d = self.config.d;
        let nh = self.config.nh;
        let nkvh = self.config.nkvh;
        let dh = d / nh;
        let dkv = nkvh * dh;
        let di = self.config.di;
        let head_group = nh / nkvh;
        let head_div = (dh as f32).sqrt().recip();

        let reusing = (d + dkv + dkv).max(di + di);
        let mut state_buf = Tensor::alloc(dt, &[nt, d + reusing], Blob::new);
        macro_rules! state {
            () => {
                split!(state_buf.as_mut().map_physical(|u| LocalSplitable::from(&mut **u)); [1]: d, reusing)
            };
        }

        let mut q_buf = Blob::new((nh * max_seq_len * dh) as usize * dt.nbytes());
        let mut att_buf = Blob::new((nh * max_seq_len * max_att_len) as usize * dt.nbytes());
        let pos = causal_lm::pos(&queries, nt);
        let pos = pos.as_ref().map_physical(|u| reslice(u));
        let mut moe_w = tensor(dt, &[nt, self.config.k]);
        let mut moe_i = tensor(U32, &[nt, self.config.k]);
        let mut routes = tensor(dt, &[nt, self.config.ne]);

        let mut x = token_embedded;
        for layer in 0..self.config.nlayers {
            let (mut x1, qkv) = state!();
            let mut qkv = qkv.slice(&[slice![=>], slice![=> d + dkv + dkv]]);

            let input_layernorm = self.params.input_layernorm(layer);
            self.kernels
                .rms_norm(&mut x1, &x, &input_layernorm, self.config.epsilon, &ThisThread);

            let w_qkv = self.params.w_qkv(layer).transpose(&[1, 0]);
            self.kernels
                .mat_mul(&mut qkv, 0., &x1, &w_qkv, 1., &ThisThread);

            let (q, k, v) = split!(qkv; [1]: d, dkv, dkv);
            let mut q = q.reshape(&[nt, nh, dh]);
            let mut k = k.reshape(&[nt, nkvh, dh]);
            let v = v.reshape(&[nt, nkvh, dh]);
            let o = x1.reshape(&[nt, nh, dh]);

            self.kernels.rope(&mut q, &pos, self.config.theta, &ThisThread);
            self.kernels.rope(&mut k, &pos, self.config.theta, &ThisThread);

            let q = q.transpose(&[1, 0, 2]).split(1, &seq_len);
            let k = k.transpose(&[1, 0, 2]).split(1, &seq_len);
            let v = v.transpose(&[1, 0, 2]).split(1, &seq_len);
            let o = o.transpose(&[1, 0, 2]).split(1, &seq_len);

            for (query, q, k, v, mut o) in izip!(&mut queries, q, k, v, o) {                
                let pos = query.pos();
                let seq_len = query.seq_len();
                let att_len = query.att_len();
                let Some((mut k_cache, mut v_cache)) = query.cache(layer as _) else {
                    continue;
                };

                let slice_cat = &[slice![=>], slice![pos =>=> seq_len], slice![=>]];
                let slice_att = &[slice![=>], slice![      => att_len], slice![=>]];
                let shape_q0 = &[nkvh * head_group, seq_len, dh];
                let shape_q1 = &[nkvh, head_group * seq_len, dh];
                let shape_att0 = &[nkvh, head_group * seq_len, att_len];
                let shape_att1 = &[nkvh * head_group, seq_len, att_len];

                let mut q_att = Tensor::new(dt, shape_q0, &mut q_buf[..]);
                let mut k_cat = k_cache.as_mut().slice(slice_cat).map_physical(|u| &mut **u);
                let mut v_cat = v_cache.as_mut().slice(slice_cat).map_physical(|u| &mut **u);
                q.reform_to(&mut q_att);
                k.reform_to(&mut k_cat);
                v.reform_to(&mut v_cat);

                let q_att = q_att.reshape(shape_q1);
                let k_att = k_cache.slice(slice_att).transpose(&[0, 2, 1]);
                let v_att = v_cache.slice(slice_att);

                let mut att = Tensor::new(dt, shape_att0, &mut att_buf[..]);
                self.kernels
                    .mat_mul(&mut att, 0., &q_att, &k_att, head_div, &ThisThread);
                let mut att = att.reshape(shape_att1);
                self.kernels.softmax(&mut att, &ThisThread);
                let mut x2 = q_att;
                self.kernels.mat_mul(
                    &mut x2,
                    0.,
                    &att.reshape(shape_att0),
                    &v_att,
                    1.,
                    &ThisThread,
                );

                x2.reshape(shape_q0).reform_to(&mut o);
            }

            let (mut x1, gate_up) = state!();
            let gate_up = gate_up.slice(&[slice![=>], slice![=> di + di]]);

            let wo = self.params.w_o(layer).transpose(&[1, 0]);
            self.kernels.mat_mul(&mut x, 1., &x1, &wo, 1., &ThisThread);

            let post_layernorm = self.params.post_attention_layernorm(layer);
            self.kernels
                .rms_norm(&mut x1, &x, &post_layernorm, self.config.epsilon, &ThisThread);

            let w_moe_gate = self.params.moe_gate(layer).transpose(&[1, 0]);
            self.kernels
                .mat_mul(&mut routes, 0., &x1, &w_moe_gate, 1., &ThisThread);
            softmax(&mut routes);
            topk(&routes, self.config.k as _, &mut moe_w, &mut moe_i);
            let weights: &[f16] = reslice(moe_w.as_slice());
            let indices: &[u32] = reslice(moe_i.as_slice());

            // x residual
            // x1 post layernorm
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
                for k in 0..self.config.k {
                    let expert = indices[(tok * self.config.k + k) as usize];
                    let expert_w = weights[(tok * self.config.k + k) as usize].to_f32() / sum;
                    let w_gate_up = self.params.mlp_gate_up(layer, expert).transpose(&[1, 0]);
                    self.kernels.mat_mul(
                        &mut gate_up_slice,
                        0.,
                        &x1_slice,
                        &w_gate_up,
                        1.,
                        &ThisThread,
                    );
                    let mut gate_up_slice = gate_up_slice.split(1, &[di as _, di as _]);
                    let up = gate_up_slice.pop_back().unwrap();
                    let mut gate = gate_up_slice.pop_back().unwrap();
                    self.kernels.swiglu(&mut gate, &up, &ThisThread);
                    let mlp_down = self.params.mlp_down(layer, expert).transpose(&[1, 0]);
                    self.kernels.mat_mul(
                        &mut x0_slice,
                        1.,
                        &gate,
                        &mlp_down,
                        expert_w,
                        &ThisThread,
                    );
                }
            }
        }

        x
    }

    fn decode(
        &self,
        decoding: impl IntoIterator<Item = DecodingMeta>,
        hidden_state: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage> {
        let dt = self.config.dtype;
        let d = self.config.d;

        let mut x = hidden_state;
        let range = DecodingMeta::select(&mut x, decoding, |dst, src| dst.copy_from_slice(src));

        if range.is_empty() {
            return Tensor::alloc(dt, &[0, d as _], Blob::new);
        }

        let lm_layernorm = &self.params.model_norm();
        let lm_head = self.params.lm_head().transpose(&[1, 0]);
        let mut x = x.slice(&[slice![range.start => range.end], slice![=>]]);
        let mut logits = Tensor::alloc(dt, &[x.shape()[0], lm_head.shape()[1]], Blob::new);

        // 复制一个 x 以实现原地归一化
        let x_ = x
            .as_ref()
            .map_physical(|u| unsafe { from_raw_parts(u.as_ptr(), u.len()) });
        self.kernels
            .rms_norm(&mut x, &x_, lm_layernorm, self.config.epsilon, &ThisThread);
        self.kernels
            .mat_mul(&mut logits, 0., &x, &lm_head, 1., &ThisThread);

        logits
    }

    fn sample(
        &self,
        args: impl IntoIterator<Item = SampleMeta>,
        logits: Tensor<Self::Storage>,
    ) -> Vec<utok> {
        let &[_, voc] = logits.shape() else { panic!() };
        let logits: &[f16] = reslice(logits.as_slice());
        args.into_iter()
            .flat_map(|meta| repeat(meta.args).take(meta.num_decode))
            .enumerate()
            .map(|(i, args)| args.random(&common_cpu::slice!(logits; voc; [i])))
            .collect()
    }
}

#[inline]
fn tensor(dt: DigitLayout, shape: &[udim]) -> Tensor<Blob> {
    Tensor::alloc(dt, shape, Blob::new)
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
        let max = line.iter().cloned().fold(f16::NEG_INFINITY, f16::max).to_f32();
        let mut exp_sum = 0.;
        for i in 0..dim {
            let val = (line[i as usize].to_f32() - max).exp();
            exp_sum += val;
        }
        for i in 0..dim {
            let val = (line[i as usize].to_f32() - max).exp();
            line[i as usize] = f16::from_f32(val / exp_sum);
        }
    }
}

#[test]
fn test_topk() {
    use digit_layout::types::{F16, U32};
    let r = 2;
    let k = 2;
    let n = 8;
    let mut blob = Blob::new(r * n * 2);
    let arr = [
        0., 2., 0., 0., 0., 0., 1., 0., 3., 0., 0., 0., 4., 0., 0., 0.,
    ];
    let src = &arr
        .iter()
        .map(|x| f16::from_f32(*x as f32))
        .collect::<Vec<_>>();
    blob.copy_from_slice(reslice(&src));
    let logits = Tensor::new(F16, &[r as u32, n as u32], blob);
    let mut weights = Tensor::alloc(F16, &[r as u32, k as u32], Blob::new);
    let mut indices = Tensor::alloc(U32, &[r as u32, k as u32], Blob::new);
    topk(&logits, k, &mut weights, &mut indices);
    let weights: &[f16] = reslice(weights.as_slice()); // [2., 1., 4., 3.]
    let indices: &[u32] = reslice(indices.as_slice()); // [1, 6, 4, 0]
    assert_eq!(weights[0], f16::from_f64(2.));
    assert_eq!(indices[0], 1);
    assert_eq!(weights[1], f16::from_f64(1.));
    assert_eq!(indices[1], 6);
    assert_eq!(weights[2], f16::from_f64(4.));
    assert_eq!(indices[2], 4);
    assert_eq!(weights[3], f16::from_f64(3.));
    assert_eq!(indices[3], 0);
}
