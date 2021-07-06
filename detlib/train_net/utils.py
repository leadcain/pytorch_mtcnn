import torch
import torch.backends.cudnn as cudnn

def check_gpus():
    print("GPU Available : {:} ".format(torch.cuda.is_available()))
    print("GPU Device : {:} ".format(torch.cuda.get_device_name()))
    print("GPU Counts : {:} ".format(torch.cuda.device_count()))
    return


def cuda_benchmark(mode="fast"):
    if mode == "fast":
        cudnn.enabled = True
        cudnn.benchmark = True
        cudnn.deterministic = False
        print("[INFO] cudnn benchmark : fast mode")

    elif mode == "deterministic":
        cudnn.enabled = False
        cudnn.benchmark = False
        cudnn.deterministic = True
        print("[INFO] cudnn benchmark : exact mode")

    return


def init_seed(seed=0):
    torch.manual_seed(seed)

    if seed == 0:
        cudnn.deterministic = False
        cudnn.benchmark = True

    return



