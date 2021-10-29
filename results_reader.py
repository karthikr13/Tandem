import os
import numpy as np
import matplotlib.pyplot as plt

def read_results(root):
    r_sum = []
    for subdirs, _, files in os.walk(root):
        for file in files:
            if file != 'results.txt':
                continue
            results = open(os.path.join(subdirs, file))
            lines = results.readlines()
            lines = [line.rstrip() for line in lines]
            sig = float((lines[2].split(": ")[1]))
            lam = float(lines[3].split(": ")[1])
            inf_err1 = float(lines[4].split("be ")[1])
            inf_err2= float(lines[5].split("be ")[1])
            inf_err = float(lines[6].split("be ")[1])
            r = (lam, sig, inf_err, inf_err1, inf_err2)
            r_sum.append(r)
            results.close()
    r_sum.sort(key=lambda r: (r[0], r[1]))
    write_file = open(os.path.join(root, "summary.csv"), "w")
    write_file.write("lambda, sig squared, inference error, model 1 error, model 2 error\n")
    for result in r_sum:
        for item in result:
            write_file.write(str(item))
            write_file.write(",")
        write_file.write("\n")
    write_file.close()

    sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    s2 = np.square(sigmas)
    plt.figure(1)
    r_sum = np.array(r_sum)
    lams = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14]

    for s in s2:
        plt.scatter(r_sum[r_sum[:, 1]==s][:,0], r_sum[r_sum[:, 1]==s][:,2], label = "$\sigma^2$={:0.2f}".format(s))
    plt.title("Inference error as a function of $\lambda$")
    plt.legend(ncol=2)
    plt.xlabel("lambda")
    plt.ylabel("Inference Error")
    plt.savefig("test/error_viz.png")

    plt.figure(2)
    for l in lams:
        plt.scatter(r_sum[r_sum[:, 0] == l][:,1], r_sum[r_sum[:, 0] == l][:,2], label="$\lambda$={:0.2f}".format(l))
    plt.title("Inference error as a function of $\sigma^2$")
    plt.legend(ncol=2)
    plt.xlabel("sigma squared")
    plt.ylabel("Inference Error")
    plt.savefig("test/error_viz_l.png")

    plt.figure(3)
    plt.clf()
    for r in r_sum:
        print(r)
        plt.scatter(r[0], r[1], c=r[2])
    print(len(r_sum))
    plt.colorbar()
    plt.xlabel('lambda')
    plt.ylabel('sigma')
    plt.savefig("test/error_viz_mse.png")
if __name__ == '__main__':
    read_results("runs-10.27")