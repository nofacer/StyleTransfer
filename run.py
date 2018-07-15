from torch import nn, optim
import torch
import Model
import util


def get_input_param_optimizer(input_img):
    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer


def run_style_transfer(content_img, style_img, input_img, num_epoches=300):
    print('Building model...')
    model, style_loss_list, content_loss_list = Model.buile_model(style_img, content_img)
    input_param, optimizer = get_input_param_optimizer(input_img)

    print('Optimizing...')
    epoch = [0]
    while epoch[0] < num_epoches:
        print(epoch[0])
        def closure():
            input_param.data.clamp_(0, 1)

            model(input_param)
            style_score = 0
            content_score = 0

            optimizer.zero_grad()
            for sl in style_loss_list:
                style_score += sl.backward()
            for cl in content_loss_list:
                content_score += cl.backward()

            epoch[0] += 1
            if epoch[0] % 50 == 0:
                print('run {}'.format(epoch))
                print('Style Loss:{:.4f} Content Loss:{:.4f}'.format(style_score.data[0], content_score.data[0]))
                print()

            return style_score + content_score

        optimizer.step(closure)
        input_param.data.clamp_(0, 1)
    return input_param.data


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_img = util.load_img('demo/content.jpg').to(device)
    style_img = util.load_img('demo/style.jpg').to(device)
    input_img = content_img.clone()
    o = run_style_transfer(content_img, style_img, input_img)
    util.show_img(o)
