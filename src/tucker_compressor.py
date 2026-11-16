import time
import numpy as np
import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import partial_tucker

tl.set_backend('pytorch')


# =============================================================================
# SELEÇÃO DE RANK POR ENERGIA (mesma heurística do TT, aplicada aos modos de canal)
# =============================================================================
def svd_with_energy(matriz, energy_threshold=0.95):
    U, S, Vh = torch.linalg.svd(matriz, full_matrices=False)
    total = torch.sum(S ** 2)
    acc = torch.cumsum(S ** 2, dim=0)
    r = torch.searchsorted(acc, energy_threshold * total).item() + 1
    return min(r, U.shape[1])

def tucker_ranks_by_energy(weight, thr):
    """Rank de cada modo de CANAL pela energia do unfolding correspondente.
    Modo 0 (C_out): unfolding [C_out, C_in*Kh*Kw]. Modo 1 (C_in): [C_in, C_out*Kh*Kw]."""
    C_out, C_in, K_h, K_w = weight.shape
    with torch.no_grad():
        r_out = svd_with_energy(weight.reshape(C_out, -1), thr)
        r_in  = svd_with_energy(weight.permute(1, 0, 2, 3).reshape(C_in, -1), thr)
    # nunca pedir rank >= dimensão (não comprime e pode inflar)
    r_out = min(r_out, C_out)
    r_in  = min(r_in,  C_in)
    return [max(1, r_out), max(1, r_in)]


# =============================================================================
# CAMADA TUCKER-2 NOS MODOS DE CANAL — FORWARD SEQUENCIAL (Kim et al., 2016)
# As 3 convoluções são nn.Conv2d REAIS e persistentes -> quantizáveis.
# =============================================================================
class TuckerConv2d(nn.Module):
    def __init__(self, layer: nn.Conv2d, ranks):
        super().__init__()
        C_out, C_in, K_h, K_w = layer.weight.shape
        r_out, r_in = ranks

        # Decomposição Tucker-2 nos modos 0 (C_out) e 1 (C_in); kernel espacial intacto
        with torch.no_grad():
            (core, factors), _ = partial_tucker(
                layer.weight.data, modes=[0, 1], rank=[r_out, r_in], init='svd')
            U_out, U_in = factors  # U_out: [C_out, r_out] | U_in: [C_in, r_in]

        # 1) REDUCE — 1x1: C_in -> r_in     (corta canais de entrada)
        self.reduce = nn.Conv2d(C_in, r_in, kernel_size=1, bias=False)
        # 2) CORE   — KxK: r_in -> r_out    (convolução cara no espaço REDUZIDO; carrega stride/pad/dil)
        self.core = nn.Conv2d(r_in, r_out, kernel_size=(K_h, K_w),
                              stride=layer.stride, padding=layer.padding,
                              dilation=layer.dilation, bias=False)
        # 3) EXPAND — 1x1: r_out -> C_out   (devolve canais; carrega o bias)
        self.expand = nn.Conv2d(r_out, C_out, kernel_size=1, bias=(layer.bias is not None))

        with torch.no_grad():
            self.reduce.weight.copy_(U_in.t().contiguous().view(r_in, C_in, 1, 1))
            self.core.weight.copy_(core.contiguous())
            self.expand.weight.copy_(U_out.contiguous().view(C_out, r_out, 1, 1))
            if layer.bias is not None:
                self.expand.bias.copy_(layer.bias)

    def forward(self, x):
        # NUNCA reconstrói o kernel denso: 3 convoluções reais, cada uma menor.
        return self.expand(self.core(self.reduce(x)))


# =============================================================================
# SUBSTITUIÇÃO RECURSIVA (mesmo filtro do TT, para comparação justa)
# =============================================================================
def replace_layers_with_tucker(model: nn.Module, energy_threshold: float = 0.95):
    for name, child in model.named_children():
        if isinstance(child, nn.Conv2d):
            if child.kernel_size == (3, 3) and child.groups == 1 and child.in_channels > 64:
                ranks = tucker_ranks_by_energy(child.weight.data, energy_threshold)
                setattr(model, name, TuckerConv2d(child, ranks))
        else:
            replace_layers_with_tucker(child, energy_threshold)
    return model


# =============================================================================
# UTILITÁRIOS DE MEDIÇÃO
# =============================================================================
def get_model_size_mb(model):
    p = sum(t.nelement() * t.element_size() for t in model.parameters())
    b = sum(t.nelement() * t.element_size() for t in model.buffers())
    return (p + b) / 1024 ** 2

@torch.no_grad()
def benchmark_latency(model, device, input_shape=(1, 3, 224, 224),
                      warmup=10, runs=50, use_amp=False):
    """Latência de inferência medida CORRETAMENTE: warmup + sincronização CUDA.
    Sem warmup, a 1ª passada inclui autotuning do cuDNN e inicialização lazy
    (foi o que inflou o baseline para 3.9s na sua tabela). Retorna ms."""
    from torch.amp import autocast
    model.eval()
    x = torch.randn(*input_shape, device=device)
    use_cuda = (device.type == 'cuda')

    for _ in range(warmup):
        with autocast('cuda', enabled=use_amp and use_cuda):
            model(x)
    if use_cuda:
        torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        if use_cuda:
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            with autocast('cuda', enabled=use_amp):
                model(x)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))   # ms
        else:
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)

    t = np.array(times)
    return dict(median_ms=float(np.median(t)), mean_ms=float(t.mean()), std_ms=float(t.std()))


def count_macs(model, device, input_shape=(1, 3, 224, 224)):
    """Conta MACs (multiply-accumulate) das convoluções via hooks de forward.
    Para Tucker, soma os 3 sub-convs -> mostra a redução real de contas.
    Para o TT (kernel denso reconstruído), contaria o conv cheio -> sem redução."""
    macs = [0]; handles = []
    def hook(mod, inp, out):
        Co = mod.out_channels; Ci = mod.in_channels // mod.groups
        Kh, Kw = mod.kernel_size
        macs[0] += Co * Ci * Kh * Kw * out.shape[2] * out.shape[3]
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(torch.randn(*input_shape, device=device))
    for h in handles:
        h.remove()
    return macs[0]

def tucker_layer_to_weight(tk):
    """Reconstrói o kernel denso de uma TuckerConv2d (só p/ medir erro de reconstrução)."""
    C_out, r_out = tk.expand.weight.shape[0], tk.expand.weight.shape[1]
    r_in, C_in = tk.reduce.weight.shape[0], tk.reduce.weight.shape[1]
    U_out = tk.expand.weight.detach().view(C_out, r_out)
    U_in  = tk.reduce.weight.detach().view(r_in, C_in).t().contiguous()
    core  = tk.core.weight.detach()
    return tl.tucker_to_tensor((core, [U_out, U_in]))
