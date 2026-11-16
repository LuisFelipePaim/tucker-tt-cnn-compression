"""
Quantização aplicada a um modelo já decomposto por Tucker e fine-tunado.
ORDEM CORRETA: Tucker -> fine-tune -> (congela) -> quantiza -> mede.

Dois caminhos:
  * fp16  : meia precisão. ~2x menor e ~2x mais rápido em GPU com tensor cores.
            Testável/roda em GPU; em CPU serve só de validação de fluxo.
  * int8  : PTQ estático via FX graph mode (fusão automática Conv-BN-ReLU).
            Rodável/acelerado em CPU (backend x86/fbgemm). Para INT8 acelerado
            em GPU, exporte para TensorRT (fora deste script).
"""
import os, time, numpy as np, torch, torch.nn as nn


def get_size_mb_disk(model, path="_tmp_q.pth"):
    torch.save(model.state_dict(), path)
    mb = os.path.getsize(path) / 1024**2
    os.remove(path)
    return mb


# -----------------------------------------------------------------------------
# fp16
# -----------------------------------------------------------------------------
def to_fp16(model):
    """Converte para meia precisão. BatchNorm é mantido em fp32 por estabilidade
    numérica (padrão recomendado); só conv/linear viram fp16."""
    model = model.half()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.float()
    return model


# -----------------------------------------------------------------------------
# int8 PTQ (FX graph mode) — CPU
# -----------------------------------------------------------------------------
def quantize_int8_ptq(model, calib_batches, example_input, backend="x86"):
    """PTQ estático. calib_batches: iterável de tensores de ENTRADA (dados de
    TREINO — nunca de teste, senão vaza). example_input: 1 tensor de exemplo."""
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

    model = model.eval().cpu()
    qmap = get_default_qconfig_mapping(backend)
    prepared = prepare_fx(model, qmap, example_inputs=(example_input.cpu(),))

    # Calibração: registra as faixas de ativação passando dados de treino
    with torch.no_grad():
        for xb in calib_batches:
            prepared(xb.cpu())

    quantized = convert_fx(prepared)   # aplica escala/zero-point e fusão
    return quantized


@torch.no_grad()
def benchmark_latency(model, device, shape=(1,3,224,224), warmup=5, runs=20, fp16=False):
    model.eval()
    x = torch.randn(*shape, device=device)
    if fp16: x = x.half()
    cuda = (device.type == "cuda")
    for _ in range(warmup): model(x)
    if cuda: torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        if cuda:
            s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
            s.record(); model(x); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
        else:
            t0=time.perf_counter(); model(x); ts.append((time.perf_counter()-t0)*1000)
    return float(np.median(ts))
