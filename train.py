"""
training file
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler

from tandem import Forward, Backward
import data_reader



def boundary_loss(x):
    """
    Used to calculate boundary loss predicted x
    :param x: predicted input tensor
    :return: boundary loss
    """

    mean = np.array([0, 0])
    range = np.array([2, 2])

    input_diff = torch.abs(x - torch.tensor(mean, dtype=torch.float))
    mean_diff = input_diff - 0.5*torch.tensor(range, dtype = torch.float)
    relu = nn.ReLU()
    total_loss = relu(mean_diff)
    return torch.mean(total_loss)

def loss(ys, labels, x = None):
    bdy = 0
    if x is not None:
        bdy = boundary_loss(x)

    mse = nn.functional.mse_loss(ys, labels)
    return bdy + mse

def train():
    train_data, test_data = data_reader.read_data_sine_wave()

    num_epochs = 500
    #num_epochs = 40
    #num_epochs = 5

    model_f = Forward()
    model_b = Backward()
    opt_f = torch.optim.Adam(model_f.parameters(), lr=0.001, weight_decay=0.0005)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=0.001)

    cuda = True if torch.cuda.is_available() else False
    if cuda:
        model_f.cuda()
        model_b.cuda()

    #Forward
    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer=opt_f, mode='min',
                                              factor=0.5,
                                              patience=10, verbose=True, threshold=1e-4)
    forward_train_losses = []
    forward_eval_losses = []
    for epoch in range(num_epochs):
        model_f.train()
        epoch_losses = []
        for i, (g, s) in enumerate(train_data):
            if cuda:
                g.cuda()
                s.cuda()

            opt_f.zero_grad()
            out = model_f(g)
            l = loss(out, s, x = None)
            l.backward()
            opt_f.step()
            epoch_losses.append(l.detach().numpy())
        epoch_loss = np.mean(epoch_losses)
        forward_train_losses.append(epoch_loss)
        lr_sched.step(epoch_loss)

        if epoch % 20 == 0:
            model_f.eval()
            print("Eval epoch " + str(epoch))
            eval_epoch_losses = []
            for i, (g, s) in enumerate(test_data):
                if cuda:
                    g.cuda()
                    s.cuda()

                out = model_f(g)
                l = loss(out, s)
                eval_epoch_losses.append(l.detach().numpy())
            epoch_loss = np.mean(eval_epoch_losses)
            forward_eval_losses.append(epoch_loss)
            print("Forward train loss on epoch {}: {}".format(epoch, forward_train_losses[-1]))
            print("Forward eval loss on epoch {}: {}".format(epoch, epoch_loss))

    #Backward
    print("Training backwards model")
    model_f.eval()
    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer=opt_b, mode='min',
                                              factor=0.5,
                                              patience=10, verbose=True, threshold=1e-4)

    backward_train_losses, backward_eval_losses = [], []
    for epoch in range(num_epochs):
        model_b.train()
        epoch_losses = []
        for i, (g, s) in enumerate(train_data):
            if cuda:
                g.cuda()
                s.cuda()

            opt_b.zero_grad()
            g_out = model_b(s)
            s_out = model_f(g_out)
            l = loss(s_out, s, x=g_out)
            l.backward()
            opt_b.step()
            epoch_losses.append(l.detach().numpy())
        epoch_loss = np.mean(epoch_losses)
        backward_train_losses.append(epoch_loss)
        lr_sched.step(epoch_loss)
        if epoch % 20 == 0:
            model_b.eval()
            print("Eval epoch " + str(epoch))
            eval_epoch_losses = []
            for i, (g, s) in enumerate(test_data):
                if cuda:
                    g.cuda()
                    s.cuda()

                g_out = model_b(s)
                s_out = model_f(g_out)
                l = loss(s_out, s, x = g_out)
                eval_epoch_losses.append(l.detach().numpy())
            epoch_loss = np.mean(eval_epoch_losses)
            backward_eval_losses.append(epoch_loss)
            print("Backwards train loss on epoch {}: {}".format(epoch, backward_train_losses[-1]))
            print("Backwards eval loss on epoch {}: {}".format(epoch, epoch_loss))

    #evaluate
    model_b.eval()
    model_f.eval()
    inference_err = 0
    for i, (g, s) in enumerate(test_data):
        print("g:")
        g_np = g.detach().numpy()
        print(g.size())
        g_out = model_b(s)
        g_out_np = g_out.detach().numpy()
        print("g_out:")
        print(g_out.size())
        s_out = np.zeros(np.array([np.shape(g_out)[0], 1]))
        s_out = np.sin(3 * np.pi * g_out_np[:,0]) + np.cos(3 * np.pi * g_out_np[:,1])
        s_out = torch.tensor(s_out, dtype=torch.float)
        s_out = torch.unsqueeze(s_out, 1)
        print(s_out.size())
        print(s.size())
        inference_err += loss(s_out, s)

    inference_err /= (i+1)

    print("Inference error found to be {}".format(inference_err))



    plt.figure(1)
    plt.title("Forward training error {:0.4f}".format(min(forward_train_losses)))
    plt.plot(range(num_epochs), forward_train_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Train Error')
    plt.yscale("log")
    plt.savefig("forward_loss_train.png")

    plt.figure(2)
    plt.title("Forward eval error {:0.4f}".format(min(forward_eval_losses)))
    plt.plot(range(0, num_epochs, 20), forward_eval_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Eval Error')
    plt.yscale("log")
    plt.savefig("forward_loss_eval.png")

    plt.figure(3)
    plt.title("Backward training error {:0.4f}".format(min(backward_train_losses)))
    plt.plot(range(num_epochs), backward_train_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Train Error')
    plt.yscale("log")
    plt.savefig("backward_loss_train.png")

    plt.figure(4)
    plt.title("Backward eval error {:0.4f}".format(min(backward_eval_losses)))
    plt.plot(range(0, num_epochs, 20), backward_eval_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Test Error')
    plt.yscale("log")
    plt.savefig("backward_loss_eval.png")

if __name__ == '__main__':
    print("Beginning training")
    train()
    print("Done training")
