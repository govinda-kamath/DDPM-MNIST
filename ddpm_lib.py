"""Shared model, noise schedule, and sampler used by train.py and sample.py."""

import jax
import jax.numpy as jnp
import equinox as eqx


# noise schedule

def make_noise_schedule(T=1000, beta_start=1e-4, beta_end=0.02):
    betas     = jnp.linspace(beta_start, beta_end, T)
    alphas    = 1.0 - betas
    alpha_bar = jnp.cumprod(alphas)
    return dict(
        betas         = betas,
        sqrt_ab       = jnp.sqrt(alpha_bar),
        sqrt_one_m_ab = jnp.sqrt(1.0 - alpha_bar),
        sqrt_recip_a  = 1.0 / jnp.sqrt(alphas),
        posterior_var = betas * (1.0 - jnp.roll(alpha_bar, 1).at[0].set(1.0))
                        / (1.0 - alpha_bar),
    )

def q_sample(x0, t, sched, key):
    """Forward diffusion: x_t = sqrt_ab_t * x0 + sqrt_one_m_ab_t * eps."""
    eps       = jax.random.normal(key, x0.shape)
    sqrt_ab   = sched['sqrt_ab'][t][:, None, None, None]
    sqrt_1mab = sched['sqrt_one_m_ab'][t][:, None, None, None]
    return sqrt_ab * x0 + sqrt_1mab * eps, eps


# model

def sinusoidal_embedding(t, dim):
    """t: scalar int → (dim,) sinusoidal embedding."""
    half  = dim // 2
    freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half) / (half - 1))
    args  = t.astype(jnp.float32) * freqs
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)])


class ResBlock(eqx.Module):
    norm1:     eqx.nn.GroupNorm
    conv1:     eqx.nn.Conv2d
    time_proj: eqx.nn.Linear
    norm2:     eqx.nn.GroupNorm
    conv2:     eqx.nn.Conv2d
    skip_conv: eqx.nn.Conv2d | None

    def __init__(self, in_ch, out_ch, emb_dim, *, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.norm1     = eqx.nn.GroupNorm(groups=8, channels=in_ch)
        self.conv1     = eqx.nn.Conv2d(in_ch, out_ch, 3, padding=1, key=k1)
        self.time_proj = eqx.nn.Linear(emb_dim, out_ch, key=k2)
        self.norm2     = eqx.nn.GroupNorm(groups=8, channels=out_ch)
        self.conv2     = eqx.nn.Conv2d(out_ch, out_ch, 3, padding=1, key=k3)
        self.skip_conv = eqx.nn.Conv2d(in_ch, out_ch, 1, key=k4) if in_ch != out_ch else None

    def __call__(self, x, t_emb):
        h    = self.conv1(jax.nn.silu(self.norm1(x)))
        h    = h + self.time_proj(jax.nn.silu(t_emb))[:, None, None]
        h    = self.conv2(jax.nn.silu(self.norm2(h)))
        skip = self.skip_conv(x) if self.skip_conv is not None else x
        return h + skip


class SmallUNet(eqx.Module):
    time_emb_dim: int = eqx.field(static=True)
    t_dense1:  eqx.nn.Linear
    t_dense2:  eqx.nn.Linear
    t_dense3:  eqx.nn.Linear
    t_dense4:  eqx.nn.Linear
    t_dense5:  eqx.nn.Linear
    init_conv: eqx.nn.Conv2d
    enc1:      ResBlock
    enc2:      ResBlock
    enc3:      ResBlock
    down:      eqx.nn.Conv2d
    enc_14:    ResBlock
    enc_14_2:  ResBlock
    mid:       ResBlock
    mid2:      ResBlock
    mid3:      ResBlock
    dec_14:    ResBlock
    dec1:      ResBlock
    dec2:      ResBlock
    dec3:      ResBlock
    norm_out:  eqx.nn.GroupNorm
    out_conv:  eqx.nn.Conv2d

    def __init__(self, base_channels=16, time_emb_dim=256, *, key):
        C = base_channels
        D = 256  # fixed projection dim; time_emb_dim controls sinusoidal basis width only
        self.time_emb_dim = time_emb_dim
        ks = iter(jax.random.split(key, 20))
        self.t_dense1  = eqx.nn.Linear(time_emb_dim, D, key=next(ks))
        self.t_dense2  = eqx.nn.Linear(D, D, key=next(ks))
        self.t_dense3  = eqx.nn.Linear(D, D, key=next(ks))
        self.t_dense4  = eqx.nn.Linear(D, D, key=next(ks))
        self.t_dense5  = eqx.nn.Linear(D, D, key=next(ks))
        self.init_conv = eqx.nn.Conv2d(1, C, 3, padding=1, key=next(ks))
        self.enc1      = ResBlock(C,     C,     D, key=next(ks))
        self.enc2      = ResBlock(C,     C,     D, key=next(ks))
        self.enc3      = ResBlock(C,     C,     D, key=next(ks))
        self.down      = eqx.nn.Conv2d(C, C*2, 3, stride=2, padding=1, key=next(ks))
        self.enc_14    = ResBlock(C*2,   C*2,   D, key=next(ks))
        self.enc_14_2  = ResBlock(C*2,   C*2,   D, key=next(ks))
        self.mid       = ResBlock(C*2,   C*2,   D, key=next(ks))
        self.mid2      = ResBlock(C*2,   C*2,   D, key=next(ks))
        self.mid3      = ResBlock(C*2,   C*2,   D, key=next(ks))
        self.dec_14    = ResBlock(C*4,   C*2,   D, key=next(ks))
        self.dec1      = ResBlock(C*2+C, C,     D, key=next(ks))
        self.dec2      = ResBlock(C,     C,     D, key=next(ks))
        self.dec3      = ResBlock(C,     C,     D, key=next(ks))
        self.norm_out  = eqx.nn.GroupNorm(8, C)
        self.out_conv  = eqx.nn.Conv2d(C, 1, 1, key=next(ks))

    def __call__(self, x, t):
        t_emb = sinusoidal_embedding(t, self.time_emb_dim)
        t_emb = self.t_dense5(jax.nn.silu(self.t_dense4(jax.nn.silu(self.t_dense3(jax.nn.silu(self.t_dense2(jax.nn.silu(self.t_dense1(t_emb)))))))))
        x  = self.init_conv(x)
        h1 = self.enc3(self.enc2(self.enc1(x, t_emb), t_emb), t_emb)
        h2 = self.enc_14_2(self.enc_14(self.down(h1), t_emb), t_emb)
        h  = self.mid(h2, t_emb)
        h  = self.mid2(h, t_emb)
        h  = self.mid3(h, t_emb)
        h  = self.dec_14(jnp.concatenate([h, h2], axis=0), t_emb)
        h  = jax.image.resize(h, (h.shape[0], 28, 28), method='nearest')
        h  = self.dec1(jnp.concatenate([h, h1], axis=0), t_emb)
        h  = self.dec2(h, t_emb)
        h  = self.dec3(h, t_emb)
        return self.out_conv(jax.nn.silu(self.norm_out(h)))


# sampler

@eqx.filter_jit
def p_sample_step(model, x_t, t_scalar, key, sched):
    """One reverse diffusion step."""
    eps_pred       = model(x_t, t_scalar)
    beta_t         = sched['betas'][t_scalar]
    sqrt_1m_ab_t   = sched['sqrt_one_m_ab'][t_scalar]
    sqrt_recip_a_t = sched['sqrt_recip_a'][t_scalar]
    post_var_t     = sched['posterior_var'][t_scalar]
    mu    = sqrt_recip_a_t * (x_t - beta_t / sqrt_1m_ab_t * eps_pred)
    noise = jax.random.normal(key, x_t.shape)
    return mu + jnp.where(t_scalar > 0, jnp.sqrt(post_var_t) * noise, 0.0)

def sample(model, n_samples, key, sched, T, shape=(1, 28, 28)):
    """Run the full reverse chain x_T -> x_0. Returns (n_samples, *shape) in [-1,1]."""
    key, init_key = jax.random.split(key)
    xs = jax.random.normal(init_key, (n_samples, *shape))
    for t_val in reversed(range(T)):
        t_scalar = jnp.array(t_val)
        keys     = jax.random.split(key, n_samples + 1)
        key      = keys[0]
        xs = jax.vmap(p_sample_step, in_axes=(None, 0, None, 0, None))(
            model, xs, t_scalar, keys[1:], sched)
    return xs
