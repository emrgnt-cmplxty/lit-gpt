import torch
import glob

if __name__ == "__main__":
    files = glob.glob("out/**/*.ckpt")
    for file in files:
        print(f"File = {file}")
        checkpoint = torch.load(file)
        state_dict = checkpoint["state_dict"]

        unwanted_prefix = "module."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        checkpoint["state_dict"] = state_dict
        torch.save(checkpoint, file)
