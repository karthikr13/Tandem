"""
training file
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import seaborn as sns
from matplotlib.colors import LogNorm
import os
from robotic_arm_data_gen import determine_final_position

from tandem import Forward, Backward
import data_reader



def boundary_loss(x):
    """
    Used to calculate boundary loss predicted x
    :param x: predicted input tensor
    :return: boundary loss
    """
    global sine
    if sine:
        mean = np.array([0, 0])
        range = np.array([2, 2])
    else:
        range, lower, upper = np.array([1.88, 3.7, 3.82, 3.78]), np.array([-0.87, -1.87, -1.92, -1.73]), np.array([1.018, 1.834, 1.897, 2.053])
        mean = (lower + upper) / 2
    if torch.cuda.is_available():
        input_diff = torch.abs(x - torch.tensor(mean, dtype=torch.float, device='cuda'))
        mean_diff = input_diff - 0.5*torch.tensor(range, dtype = torch.float, device='cuda')
    else:
        input_diff = torch.abs(x - torch.tensor(mean, dtype=torch.float))
        mean_diff = input_diff - 0.5 * torch.tensor(range, dtype=torch.float)
    relu = nn.ReLU()
    total_loss = relu(mean_diff)
    if torch.cuda.is_available():
        total_loss = total_loss.cuda()
    return torch.mean(total_loss)

def loss(ys, labels, x = None, prev = None, sigma = 1, l = 0):
    global sine
    bdy = 0
    prev_loss = 0
    if x is not None:
        bdy = boundary_loss(x)
        if prev is not None and len(prev) > 0:
            for r in range(len(prev)):
                prev_loss += torch.mean(l*(1-(torch.exp(-1 * torch.square(torch.subtract(prev[r], x)) / (sigma ** 2)))))
    mse = nn.functional.mse_loss(ys, labels)

    return 100*bdy + mse - prev_loss

def train_multiple(k, dir, lam, sigma, in_size, out_size, sin=True):
    global sine
    sine = sin
    if sin:
        train_data, test_data = data_reader.read_data_sine_wave()
    else:
        train_data, test_data = data_reader.read_data_robotic_arm()

    cuda = True if torch.cuda.is_available() else False

    num_epochs = 500
    #num_epochs = 40
    #num_epochs = 10
    #num_epochs = 2

    #set up results file
    results = open("{}/results.txt".format(dir), "w")
    results.write("Number of epochs: {}\n".format(num_epochs))
    results.write("Sigma: {}\n".format(sigma))
    results.write("Sigma squared: {}\n".format(sigma ** 2))
    results.write("Lambda: {}\n".format(lam))

    # train forward model
    print("training forward model")
    model_f = Forward(in_size, out_size)
    if cuda:
        model_f.cuda()
    opt_f = torch.optim.Adam(model_f.parameters(), lr=0.001, weight_decay=0.0005)

    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer=opt_f, mode='min',
                                              factor=0.5,
                                              patience=10, verbose=True, threshold=1e-4)
    forward_train_losses = []
    forward_eval_losses = []
    best_err_f = 20
    for epoch in range(num_epochs):
        model_f.train()
        epoch_losses = []
        for i, (g, s) in enumerate(train_data):
            if cuda:
                g = g.cuda()
                s = s.cuda()

            opt_f.zero_grad()
            out = model_f(g)
            l = loss(out, s, x=None)
            l.backward()
            opt_f.step()
            epoch_losses.append(l.cpu().detach().numpy())
        epoch_loss = np.mean(epoch_losses)
        forward_train_losses.append(epoch_loss)
        lr_sched.step(epoch_loss)

        if epoch % 20 == 0:
            model_f.eval()
            print("Eval epoch " + str(epoch))
            eval_epoch_losses = []
            for i, (g, s) in enumerate(test_data):
                if cuda:
                    g = g.cuda()
                    s = s.cuda()

                out = model_f(g)
                l = loss(out, s)
                eval_epoch_losses.append(l.cpu().detach().numpy())
            epoch_loss = np.mean(eval_epoch_losses)
            if epoch_loss < best_err_f:
                best_err_f = epoch_loss
                torch.save(model_f.state_dict(), "{}/modelf.pt".format(dir))

            forward_eval_losses.append(epoch_loss)
            print("Forward train loss on epoch {}: {}".format(epoch, forward_train_losses[-1]))
            print("Forward eval loss on epoch {}: {}".format(epoch, epoch_loss))

    model_f.eval()
    #train backwards models
    model_b = []
    opts = []
    lr_scheds = []
    inference_errs = []
    for idx in range(k):
        print("setting up backwards model {}".format(idx+1))
        model = Backward(in_size, out_size)
        if cuda:
            model.cuda()
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        opts.append(opt)
        model_b.append(model)
        lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer=opt, mode='min',
                                                  factor=0.5,
                                                  patience=10, verbose=True, threshold=1e-4)
        lr_scheds.append(lr_sched)

        print("training backwards model {} of {}".format(idx+1, k))

        best_err = float('inf')
        backward_train_losses, backward_eval_losses = [], []
        backward_train_losses2, backward_eval_losses2 = [], []

        for n in range(0, idx):
            model_b[n].eval()

        for epoch in range(num_epochs):
            model_b[idx].train()
            epoch_losses = []
            epoch_losses_mses = []
            for i, (g, s) in enumerate(train_data):
                if cuda:
                    g = g.cuda()
                    s = s.cuda()

                opts[idx].zero_grad()
                g_out = model_b[idx](s)
                s_out = model_f(g_out)
                prev = []
                for p_idx in range(0, idx):
                    prev.append(model_b[p_idx](s))
                l = loss(s_out, s, x=g_out, prev=prev, sigma=sigma, l=lam)
                l2 = loss(s_out, s, x=g_out)
                l.backward()
                opts[idx].step()
                epoch_losses.append(l.cpu().detach().numpy())
                epoch_losses_mses.append(l2.cpu().detach().numpy())
            epoch_loss = np.mean(epoch_losses)
            epoch_loss_mse = np.mean(epoch_losses_mses)
            backward_train_losses.append(epoch_loss)
            backward_train_losses2.append(epoch_loss_mse)
            lr_scheds[idx].step(epoch_loss)

            if epoch % 20 == 0:
                model_b[idx].eval()
                print("Eval epoch " + str(epoch))
                eval_epoch_losses = []
                eval_epoch_losses_mse = []
                for i, (g, s) in enumerate(test_data):
                    if cuda:
                        g = g.cuda()
                        s = s.cuda()

                    g_out = model_b[idx](s)
                    s_out = model_f(g_out)
                    prev = []
                    for p_idx in range(0, idx):
                        prev.append(model_b[p_idx](s))
                    l = loss(s_out, s, x=g_out, prev=prev, sigma=sigma, l=lam)
                    l2 = loss(s_out, s, x=g_out)
                    eval_epoch_losses.append(l.cpu().detach().numpy())
                    eval_epoch_losses_mse.append(l2.cpu().detach().numpy())
                eval_epoch_loss = np.mean(eval_epoch_losses)
                epoch_loss_mse = np.mean(eval_epoch_losses_mse)
                backward_eval_losses.append(epoch_loss)
                backward_eval_losses2.append(epoch_loss_mse)
                print("Backwards {} train loss on epoch {}: {}".format(idx + 1, epoch, backward_train_losses2[-1]))
                print("Backwards {} eval loss on epoch {}: {}".format(idx + 1, epoch, epoch_loss))
                if eval_epoch_loss < best_err:
                    best_err = eval_epoch_loss
                    torch.save(model_b[idx].state_dict(), "{}/model_b{}.pt".format(dir, idx))

        print("done training backwards model {}".format(idx+1))

        inf_err = []
        for i, (g, s) in enumerate(test_data):
            if torch.cuda.is_available():
                g = g.cuda()
                s = s.cuda()
            g_out = model_b[idx](s)
            g_out_np = g_out.cpu().detach().numpy()

            if sin:
                s_out = np.zeros(np.array([np.shape(g_out)[0], 1]))
                s_out = np.sin(3 * np.pi * g_out_np[:, 0]) + np.cos(3 * np.pi * g_out_np[:, 1])
            else:
                s_out, positions = determine_final_position(g_out_np[:, 0], g_out_np[:, 1:], evaluate_mode=True)
            s_out = torch.tensor(s_out, dtype=torch.float)
            if sin:
                s_out = torch.unsqueeze(s_out, 1)
            if torch.cuda.is_available():
                s_out = s_out.cuda()
            inf_err.append(loss(s_out, s).cpu().detach().numpy())
        inference_errs.append(np.mean(inf_err))
        print("Inference error for backwards model {}: {}".format(idx+1, np.mean(inf_err)))
        results.write("Inference error for backwards model {}: {}\n".format(idx+1, np.mean(inf_err)))

    overall_inf_err = []
    for i, (g, s) in enumerate(test_data):
        if torch.cuda.is_available():
            g = g.cuda()
            s = s.cuda()

        gs, inf_errs = [], []
        for n in range(k):
            g_out = model_b[n](s)
            g_out_np = g_out.cpu().detach().numpy()
            gs.append(g_out_np)
        for g_out in gs:
            if sin:
                s_out = np.zeros(np.array([np.shape(g_out)[0], 1]))
                s_out = np.sin(3 * np.pi * g_out[:, 0]) + np.cos(3 * np.pi * g_out[:, 1])
            else:
                s_out, positions = determine_final_position(g_out[:, 0], g_out[:, 1:], evaluate_mode=True)
            s_out = torch.tensor(s_out, dtype=torch.float)
            if sin:
                s_out = torch.unsqueeze(s_out, 1)
            if torch.cuda.is_available():
                s_out = s_out.cuda()
            inf_err = loss(s_out, s).cpu().detach().numpy()
            inf_errs.append(inf_err)
            print(inf_err)

        best = np.argmin(inf_errs)
        print(best, inf_errs)
        overall_inf_err.append(inf_errs[best])
    final_err = np.mean(overall_inf_err)
    print("Overall error: {}".format(final_err))
    results.write("Overall error: {}\n".format(final_err))
    results.close()

    if sin:
        # geometry visualization
        plt.figure(1)
        plt.clf()
        plt.title("Geometry visualization")
        test_s = np.linspace(-1, 1, 100)
        s_in = torch.tensor(test_s, dtype=torch.float).unsqueeze(1)
        if torch.cuda.is_available():
            s_in = s_in.cuda()
        for n in range(k):
            g = model_b[n](s_in)
            test_g = g.cpu().detach().numpy()
            plt.scatter(test_g[:, 0], test_g[:, 1], s=10, label='Inverse Model {}'.format(n+1))
        plt.legend()
        plt.savefig("{}/geometry_visualization.png".format(dir))

def inference(model_f, model_b, dir):
    train_data, test_data = data_reader.read_data_sine_wave()
    model_f.eval()
    for model in model_b:
        model.eval()
    print("Starting inference")

    plt.figure(1)
    plt.clf()
    plt.title("Geometry visualization")


    plt.figure(2)
    plt.clf()
    #plt.xlim([-1, 1])
    #plt.ylim([-1, 1])
    plt.title("Visualization of all inverse models with spectra predictions")

    plt.figure(3)
    plt.clf()
    plt.title("Visualization of all inverse model errors with simulator error")
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    mesh = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    mesh_ys = np.sin(3 * np.pi * mesh[0]) + np.cos(3 * np.pi * mesh[1])
    plt.pcolormesh(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), mesh_ys, shading='auto',cmap=plt.get_cmap('gray'))
    markers = ['o', 'x', 'd', '*']
    test_s = np.linspace(-1, 1, 100)
    s_in = torch.tensor(test_s, dtype=torch.float).unsqueeze(1)


    if torch.cuda.is_available():
        s_in = s_in.cuda()
    for n in range(len(model_b)):
        g = model_b[n](s_in)
        test_g = g.cpu().detach().numpy()
        true_s = np.sin(3 * np.pi * test_g[:, 0]) + np.cos(3 * np.pi * test_g[:, 1])
        err = (true_s - test_s) ** 2
        plt.figure(1)
        plt.scatter(test_g[:, 0], test_g[:, 1], s=10, label='Inverse Model {}'.format(n + 1))

        plt.figure(2)
        plt.scatter(test_g[:, 0], test_g[:, 1], s=20, c=test_s, marker = markers[n], label='Inverse Model {}'.format(n+1))

        plt.figure(3)
        plt.scatter(test_g[:, 0], test_g[:, 1], s=20, c=np.sqrt(err), marker=markers[n],
                    label='Inverse Model {}'.format(n + 1))

        plt.figure(100 + n)
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.title("Visualization of model {} error with simulator error".format(n+1))
        plt.scatter(test_g[:, 0], test_g[:, 1], s=20, c=np.sqrt(err), label='Inverse Model {}'.format(n + 1))
        plt.colorbar(label='sqrt error')
        plt.savefig("{}/err_viz_model{}.png".format(dir, n+1))

        plt.figure(200 + n)
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.title("Visualization of model {} with simulator error and predictions".format(n + 1))
        plt.pcolormesh(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), mesh_ys, shading='auto',
                       cmap=plt.get_cmap('gray'))
        plt.scatter(test_g[:, 0], test_g[:, 1], s=20, c=np.sqrt(err))
        plt.colorbar(label='sqrt error')
        plt.clim(0, 0.9)
        plt.savefig("{}/err_viz_preds_model{}.png".format(dir, n + 1))

    plt.figure(1)
    plt.legend()
    plt.savefig("{}/geometry_visualization.png".format(dir))

    plt.figure(2)
    plt.legend()
    plt.colorbar(label='corresponding spectra value')
    plt.savefig("{}/geometry_visualization_predictions.png".format(dir))

    plt.figure(3)
    plt.legend()
    plt.colorbar(label='sqrt error')
    plt.clim(0, 0.9)
    plt.savefig("{}/geometry_visualization_errors.png".format(dir))



if __name__ == '__main__':
    '''
    os.makedirs("test_mult3", exist_ok=True)
    ks = [1, 2, 3, 4]
    for i in range(1, 8):
        for k in ks:
            root = "test_mult3/k={}".format(k)
            os.makedirs(root, exist_ok=True)
            #for i in range(3):
            dir = "{}/run{}".format(root, i)
            os.makedirs(dir, exist_ok=True)
            train_multiple(k, dir, 0.02, 0.2, 2, 1)
    '''
    os.makedirs('robot_test', exist_ok=True)
    '''
    os.makedirs('robot_test/k=2', exist_ok=True)
    train_multiple(2, "robot_test/k=2", 0, 0.2, 4, 2, sin=False)
    '''
    dirs = ["robot_test/run{}".format(i+1) for i in range(5, 10)]
    for dir in dirs:
        for i in range(1,5):
            d = "{}/k={}".format(dir, i)
            os.makedirs(d, exist_ok=True)
            train_multiple(i, d, 0, 0.2, 4, 2, sin=False)
            
    '''
    os.makedirs("test_mult_viz2", exist_ok=True)
    #train_multiple(2, "test_mult2", 0.02, 0.2)
    dir = "test_mult2/k=4/run3"
    model_f = Forward()
    model_f.load_state_dict(torch.load("{}/modelf.pt".format(dir), map_location=torch.device('cpu')))

    model = Backward()
    model.load_state_dict(torch.load("{}/model_b0.pt".format(dir), map_location=torch.device('cpu')))

    model_b2 = Backward()
    model_b2.load_state_dict(torch.load("{}/model_b1.pt".format(dir), map_location=torch.device('cpu')))

    model_b3 = Backward()
    model_b3.load_state_dict(torch.load("{}/model_b2.pt".format(dir), map_location=torch.device('cpu')))

    model_b4 = Backward()
    model_b4.load_state_dict(torch.load("{}/model_b3.pt".format(dir), map_location=torch.device('cpu')))

    model_b = [model, model_b2, model_b3, model_b4]

    inference(model_f, model_b, "test_mult_viz2")
    '''