import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tensorly as tl
from tensorly.decomposition import tensor_train

tl.set_backend('pytorch')

def svd_with_energy(matriz: torch.Tensor, energy_threshold: float = 0.95):
    U, S, Vh = torch.linalg.svd(matriz, full_matrices=False)

    energia_total = torch.sum(S ** 2)
    energia_acumulada = torch.cumsum(S ** 2, dim=0)

    rank_desejado = torch.searchsorted(energia_acumulada, energy_threshold * energia_total).item() + 1

    rank_final = min(rank_desejado, U.shape[1])

    return U[:, :rank_final], S[:rank_final], Vh[:rank_final, :], rank_final


def discover_tt_ranks_by_energy(weight: torch.Tensor, energy_threshold: float) -> list:
    C_out, C_in, K_h, K_w = weight.shape

    with torch.no_grad():

        mat1 = weight.reshape(C_out, -1)
        _, S1, Vh1, r1 = svd_with_energy(mat1, energy_threshold)
        C = S1.unsqueeze(1) * Vh1


        mat2 = C.reshape(r1 * C_in, -1)
        _, S2, Vh2, r2 = svd_with_energy(mat2, energy_threshold)
        C = S2.unsqueeze(1) * Vh2


        mat3 = C.reshape(r2 * K_h, -1)
        _, _, _, r3 = svd_with_energy(mat3, energy_threshold)

    return [1, r1, r2, r3, 1]


class TTConv2dTensorly(nn.Module):
    def __init__(self, layer: nn.Conv2d, tt_ranks: list):
        super().__init__()

        self.stride = layer.stride
        self.padding = layer.padding
        self.dilation = layer.dilation
        self.groups = layer.groups

        weight = layer.weight.detach()

        with torch.no_grad():
            tt_cores = tensor_train(weight, rank=tt_ranks)
        self.core_1 = nn.Parameter(tt_cores[0])
        self.core_2 = nn.Parameter(tt_cores[1])
        self.core_3 = nn.Parameter(tt_cores[2])
        self.core_4 = nn.Parameter(tt_cores[3])

        if layer.bias is not None:
            self.bias = nn.Parameter(layer.bias.clone())
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        W_recon = tl.tt_to_tensor([self.core_1, self.core_2, self.core_3, self.core_4])
        return F.conv2d(x, W_recon, self.bias, self.stride, self.padding, self.dilation, self.groups)


def replace_layers_with_tt_tensorly(model: nn.Module, energy_threshold: float = 0.95):
    for name, child in model.named_children():

        if isinstance(child, nn.Conv2d):
            if child.kernel_size == (3, 3) and child.groups == 1 and child.in_channels > 64:

                dynamic_ranks = discover_tt_ranks_by_energy(child.weight.detach(), energy_threshold)

                new_tt_layer = TTConv2dTensorly(child, tt_ranks=dynamic_ranks)
                setattr(model, name, new_tt_layer)

                print(f"[Success] Replaced: {name} | Energy: {energy_threshold} | Ranks: {dynamic_ranks[1:-1]}")

        else:
            replace_layers_with_tt_tensorly(child, energy_threshold)

    return model

def recalibrate_bn(model: nn.Module, train_dataloader: DataLoader, num_batches: int = 100, device: str = 'cuda'):
    print("\n--- Iniciando Recalibração do Batch Normalization ---")
    model.to(device)
    model.train()

    with torch.no_grad():
        for i, (images, _) in enumerate(train_dataloader):
            if i >= num_batches:
                break

            images = images.to(device)
            _ = model(images)

            if (i+1) % 20 == 0:
                print(f"Calibrando... Lote {i+1}/{num_batches}")

    model.eval()
    print("[Success] Batch Normalization recalibrado para os novos tensores!\n")
    return model

def calculate_compression_ratio(layer_original: nn.Conv2d, layer_tt: TTConv2dTensorly):
    params_originais = layer_original.weight.numel()

    params_tt = (layer_tt.core_1.numel() +
                 layer_tt.core_2.numel() +
                 layer_tt.core_3.numel() +
                 layer_tt.core_4.numel())

    taxa_compressao = params_originais / params_tt
    espaco_economizado = (1 - (params_tt / params_originais)) * 100

    print("=== Relatório de Tamanho Tensor-Train ===")
    print(f"Parâmetros Originais: {params_originais:,}")
    print(f"Parâmetros TT:        {params_tt:,}")
    print(f"Taxa de Compressão:   {taxa_compressao:.2f}x menor")
    print(f"Espaço Economizado:   {espaco_economizado:.2f}%\n")

    return taxa_compressao, espaco_economizado

def calculate_tt_error(layer_original: nn.Conv2d, layer_tt: TTConv2dTensorly):
    weight_original = layer_original.weight.detach()

    cores = [layer_tt.core_1.detach(), layer_tt.core_2.detach(), layer_tt.core_3.detach(), layer_tt.core_4.detach()]
    weight_reconstructed = tl.tt_to_tensor(cores)

    diff = weight_original - weight_reconstructed

    norma_diff = torch.linalg.norm(diff)
    norma_original = torch.linalg.norm(weight_original)
    erro_relativo = (norma_diff / norma_original) * 100
    mse = torch.mean(diff ** 2)

    print("=== Relatório de Degradação (Erro de Reconstrução) ===")
    print(f"Erro Relativo (Frobenius): {erro_relativo.item():.2f}%")
    print(f"MSE:                       {mse.item():.6f}\n")

    return erro_relativo.item(), mse.item()

def get_model_size_mb(model: nn.Module) -> float:
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_all_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_all_mb
