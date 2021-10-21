"""
training file
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import math

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

def loss(ys, labels, x = None, prev = None):
    bdy = 0
    prev_loss = 0
    sigma = 0.5
    l = 0.02
    if x is not None:
        bdy = boundary_loss(x)
    if prev is not None:
        prev_loss = torch.mean(l*(1-(torch.exp(-1 * torch.square(torch.subtract(prev, ys)) / (sigma ** 2)))))
    mse = nn.functional.mse_loss(ys, labels)

    return bdy + mse - prev_loss

def train():
    train_data, test_data = data_reader.read_data_sine_wave()

    num_epochs = 500
    #num_epochs = 40
    #num_epochs = 5

    model_f = Forward()
    model_b = Backward()
    model_b2 = Backward()
    opt_f = torch.optim.Adam(model_f.parameters(), lr=0.001, weight_decay=0.0005)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=0.001)
    opt_b2 = torch.optim.Adam(model_b2.parameters(), lr=0.001)

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


    #Backward model 2
    backward_train_losses2, backward_eval_losses2 = [], []
    print("Training second backwards model")
    model_f.eval()
    model_b.eval()
    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer=opt_b2, mode='min',
                                              factor=0.5,
                                              patience=10, verbose=True, threshold=1e-4)
    for epoch in range(num_epochs):
        model_b2.train()
        model_b.eval()
        epoch_losses = []
        for i, (g, s) in enumerate(train_data):
            if cuda:
                g.cuda()
                s.cuda()

            opt_b2.zero_grad()
            g_out = model_b2(s)
            s_out = model_f(g_out)
            l = loss(s_out, s, x=g_out, prev=model_b(s))
            l.backward()
            opt_b2.step()
            epoch_losses.append(l.detach().numpy())
        epoch_loss = np.mean(epoch_losses)
        backward_train_losses2.append(epoch_loss)
        lr_sched.step(epoch_loss)
        if epoch % 20 == 0:
            model_b2.eval()
            print("Eval epoch " + str(epoch))
            eval_epoch_losses = []
            for i, (g, s) in enumerate(test_data):
                if cuda:
                    g.cuda()
                    s.cuda()

                g_out = model_b2(s)
                s_out = model_f(g_out)
                l = loss(s_out, s, x=g_out, prev=model_b(s))
                eval_epoch_losses.append(l.detach().numpy())
            epoch_loss = np.mean(eval_epoch_losses)
            backward_eval_losses2.append(epoch_loss)
            print("Backwards 2 train loss on epoch {}: {}".format(epoch, backward_train_losses2[-1]))
            print("Backwards 2 eval loss on epoch {}: {}".format(epoch, epoch_loss))

    #inference
    print("Starting inference")
    model_b.eval()
    model_b2.eval()
    model_f.eval()
    inference_err = []
    inference_err2 = []
    fwd_preds = []
    fwd_preds2 = []
    sim_preds = []
    sim_preds2 = []
    true_s = []
    fwd_mses = []
    sim_mses = []
    fwd_best_preds, sim_best_preds = [],[]
    for i, (g, s) in enumerate(test_data):
        g_out = model_b(s)
        g_out2 = model_b2(s)
        s_out = model_f(g_out)
        s_out2 = model_f(g_out2)
        fwd_preds += s_out.detach().numpy().tolist()
        fwd_preds2 += s_out2.detach().numpy().tolist()
        true_s += s.detach().numpy().tolist()

    for i, (g, s) in enumerate(test_data):
        print(s.size())
        g_out = model_b(s)
        g_out_np = g_out.detach().numpy()

        s_out = np.zeros(np.array([np.shape(g_out)[0], 1]))
        s_out = np.sin(3 * np.pi * g_out_np[:,0]) + np.cos(3 * np.pi * g_out_np[:,1])
        s_out = torch.tensor(s_out, dtype=torch.float)
        s_out = torch.unsqueeze(s_out, 1)

        g_out2 = model_b2(s)
        g_out_np2 = g_out2.detach().numpy()
        s_out2 = np.zeros(np.array([np.shape(g_out2)[0], 1]))
        s_out2 = np.sin(3 * np.pi * g_out_np2[:, 0]) + np.cos(3 * np.pi * g_out_np2[:, 1])
        s_out2 = torch.tensor(s_out2, dtype=torch.float)
        s_out2 = torch.unsqueeze(s_out2, 1)

        sim_preds += s_out.detach().numpy().tolist()
        sim_preds2 += s_out2.detach().numpy().tolist()

        inference_err.append(loss(s_out, s))
        inference_err2.append(loss(s_out2, s))

    for k in range(len(true_s)):
        f_p = [fwd_preds[k][0], fwd_preds2[k][0]]
        fwd_model_err = abs(fwd_preds[k][0] - true_s[k][0]) ** 2
        fwd_model_err2 = abs(fwd_preds2[k][0] - true_s[k][0]) ** 2
        best_fwd_err = min(fwd_model_err, fwd_model_err2)
        fwd_mses.append(best_fwd_err)
        fwd_best_preds.append(f_p[np.argmin([fwd_model_err, fwd_model_err2])])

        s_p = [sim_preds[k][0], sim_preds2[k][0]]
        sim_model_err = abs(sim_preds[k][0] - true_s[k][0]) ** 2
        sim_model_err2 = abs(sim_preds2[k][0] - true_s[k][0]) ** 2
        best_sim_err = min(sim_model_err, sim_model_err2)
        sim_mses.append(best_sim_err)
        sim_best_preds.append(s_p[np.argmin([sim_model_err, sim_model_err2])])
        sim_mses.append(best_sim_err)

    #results
    results = open("results.txt", "w")

    x = [np.log(fwd_mses), np.log(sim_mses)]
    inference_err_avg = np.mean(inference_err)

    print("Inference error of inverse model 1 found to be {}".format(inference_err_avg))
    print("Inference error of inverse model 2 found to be {}".format(np.mean(inference_err2)))
    print("Inference error of overall model found to be {}".format(np.mean(sim_mses)))

    results.write("Inference error of inverse model 1 found to be {}\n".format(inference_err_avg))
    results.write("Inference error of inverse model 2 found to be {}\n".format(np.mean(inference_err2)))
    results.write("Inference error of overall model found to be {}\n".format(np.mean(sim_mses)))

    #histogram
    plt.figure(1)
    plt.title("Error histogram")
    plt.xlabel("Error (10$^x$)")
    plt.ylabel("Count")
    plt.hist(x, bins=100, label=['forward', 'simulator'])
    plt.legend()
    plt.savefig("histogram.png")

    #geometry visualization
    test_s = np.linspace(-1, 1, 500)
    test_g = model_b(torch.tensor(test_s, dtype=torch.float).unsqueeze(1)).detach().numpy()
    test_g2 = model_b2(torch.tensor(test_s, dtype=torch.float).unsqueeze(1)).detach().numpy()

    plt.figure(2)
    plt.title("Visualization of output geometries")
    plt.plot(test_g[:, 0], test_g[:, 1], label='Inverse Model 1')
    plt.plot(test_g2[:, 0], test_g2[:, 1], label='Inverse Model 2')
    plt.legend()
    plt.savefig("geometry_visualization.png")

    #training graphs
    plt.figure(3)
    plt.title("Forward training error")
    results.write("Forward training error {:0.4f}\n".format(min(forward_train_losses)))
    plt.plot(range(num_epochs), forward_train_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Train Error')
    plt.yscale("log")
    plt.savefig("forward_loss_train.png")

    plt.figure(4)
    plt.title("Forward eval error")
    results.write("Forward eval error {:0.4f}\n".format(min(forward_eval_losses)))
    plt.plot(range(0, num_epochs, 20), forward_eval_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Eval Error')
    plt.yscale("log")
    plt.savefig("forward_loss_eval.png")

    plt.figure(5)
    plt.title("Backward training error")
    results.write("Backward model 1 training error {:0.4f}\n".format(min(backward_train_losses)))
    results.write("Backward model 2 training error {:0.4f}\n".format(min(backward_train_losses2)))
    plt.plot(range(num_epochs), backward_train_losses, label = 'Inverse Model 1')
    plt.plot(range(num_epochs), backward_train_losses2, label='Inverse Model 2')
    plt.xlabel('Epochs')
    plt.ylabel('Train Error')
    plt.yscale("log")
    plt.savefig("backward_loss_train.png")

    plt.figure(6)
    plt.title("Backward eval error")
    results.write("Backward model 1 eval error {:0.4f}\n".format(min(backward_eval_losses)))
    results.write("Backward model 2 eval error {:0.4f}\n".format(min(backward_eval_losses2)))
    plt.plot(range(0, num_epochs, 20), backward_eval_losses, label = 'Inverse Model 1')
    plt.plot(range(0, num_epochs, 20), backward_eval_losses2, label='Inverse Model 2')
    plt.xlabel('Epochs')
    plt.ylabel('Test Error')
    plt.yscale("log")
    plt.savefig("backward_loss_eval.png")

    results.close()
if __name__ == '__main__':
    print("Beginning training")
    train()
    print("Done training")
