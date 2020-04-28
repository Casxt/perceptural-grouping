import torch

from models import subsequent_mask
from .make_model import make_image_model
from .model import EncoderDecoder


def run(model: EncoderDecoder, src, max_len):
    memory = model.encode(src, None)
    print("memory size:", memory.shape)
    output = torch.zeros(1, 1, 784).type_as(src.data)
    for i in range(max_len - 1):
        print("decode pos:", i)
        out = model.decode(memory, None,
                           output,
                           subsequent_mask(output.size(1)).type_as(src.data))
        # 取出预测序列的最后一个元素
        out = out[:, -1:, :]
        out = model.generator(out)
        output = torch.cat([output, out], dim=1)
    return output


if __name__ == "__main__":
    # with torch.no_grad():
    # inp[:, 0, 64]是全零的开始标志
    inp = torch.zeros(1, 785, 64, dtype=torch.float)
    model = make_image_model(input_channel=inp.size(2), output_channel=784, max_len=784 * 512)
    model.eval()
    out = run(model, inp, inp.size(1))
    print("output size:", out.shape)
