import torch


def main() -> None:
    x = torch.randn(4096, 4096, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("done", float(y.norm()))


if __name__ == "__main__":
    main()