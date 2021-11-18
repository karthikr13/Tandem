import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import csv

def mult_results(root):
    runs = []
    for i in range(1,11):
        runs.append("run{}".format(i))
    out = {}
    for subdirs, _, files in os.walk(root):
        spl = subdirs.split("/")
        if len(spl) < 3:
            continue
        k = float(spl[1][-1])
        if k not in out:
            out[k] = []
        results = open("{}/results.txt".format(subdirs))
        lines = results.readlines()
        err = float(lines[-1].split(": ")[1])
        out[k].append(err)

    means = []
    plt.figure(1)
    plt.clf()
    plt.xticks(range(1, len(out.keys())+1))
    plt.xlabel('k')
    plt.ylabel('Inference error')
    plt.title("Inference error across 10 sets of k models, $\lambda$=0.2, $\sigma^2$=0.04")

    for k in range(1, len(out.keys()) + 1):
        means.append(np.mean(out[k]))
        plt.scatter([k]*len(out[k]), out[k])
    plt.plot(sorted(out.keys()), means, label='mean error')
    plt.legend()
    plt.savefig("{}/mult_sum.png".format(root))

def read_results(dirs, savedir):
    results_dict = {}
    s2 = set()
    lams = set()
    errs = []
    #for dir in dirs:

    for i in range(10):
        with open(os.path.join(dirs, 'summary{}.csv'.format(i))) as f:
            reader = csv.reader(f)
            next(reader)
            for line in reader:
                if float(line[0]) == 0:
                    continue
                key = line[0] + "," + str(round(float(line[1]), 2))
                s2.add(str(round(float(line[1]), 2)))
                lams.add(line[0])
                if key not in results_dict:
                    results_dict[key] = []
                    #results_dict[key] = 0
                #results_dict[key] += float(line[2])
                #results_dict[key] = min(float(line[2]), results_dict[key])
                results_dict[key].append(float(line[2]))
    os.makedirs(savedir, exist_ok=True)
    write_file = open("{}/full_summary.csv".format(savedir), "w")
    write_file.write("lambda, sigma squared, inference error\n")

    for key in results_dict:
        '''
        max_ind = np.argmax(results_dict[key])
        results_dict[key] = np.delete(results_dict[key], max_ind)
        min_ind = np.argmax(results_dict[key])
        results_dict[key] = np.delete(results_dict[key], min_ind)

        results_dict[key] = np.mean(results_dict[key])
        '''
        results_dict[key] = min(results_dict[key])
        #results_dict[key] /= len(dirs)
        l, s = key.split(",")
        write_file.write(l)
        write_file.write(",")
        write_file.write(s)
        write_file.write(",")
        write_file.write(str(results_dict[key]))
        write_file.write("\n")
        errs.append(results_dict[key])
    write_file.close()



    #fl_lam = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14]
    #fl_s2 = [0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81, 1]
    print(results_dict.keys())
    #fl_s2 = [0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81, 1]
    #fl_lam = [0.02]
    fl_s2 = [0.04, 0.16, 0.36, 0.64, 1]
    fl_lam = [0.02,  0.06, 0.1, 0.14]
    sns_in = []
    for s in fl_s2:
        add = []
        for l in fl_lam:
            if s == 1:
                key = "{},1.0".format(l)
            else:
                key = "{},{}".format(l, s)
            add.append(results_dict[key])
        sns_in.append(np.array(add))
    sns_in = np.array(sns_in)

    points = np.meshgrid(fl_lam, fl_s2)
    plt.scatter(points[0], points[1], c=errs)
    plt.title("Error")
    plt.xlabel("lambda")
    plt.ylabel("sigma squared")
    plt.colorbar()
    plt.savefig("{}/error.png".format(savedir))

    plt.figure(4)

    ax = sns.heatmap(sns_in, cmap='YlGnBu', norm=LogNorm(), cbar_kws={'label': 'log error'})
    ax.set_xticklabels(fl_lam)
    ax.set_yticklabels(fl_s2)
    plt.title("Error heatmap using min of 10 trials")

    plt.xlabel("$\lambda$")
    plt.ylabel("$\sigma^2$")
    plt.savefig("{}/heatmap.png".format(savedir))

    plt.figure(1)
    for key in results_dict:
        l, s2 = key.split(",")
        plt.scatter(l, s2)

def collate_results(root):
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
    '''
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
    plt.savefig("{}/error_viz.png".format(root))

    plt.figure(2)
    for l in lams:
        plt.scatter(r_sum[r_sum[:, 0] == l][:,1], r_sum[r_sum[:, 0] == l][:,2], label="$\lambda$={:0.2f}".format(l))
    plt.title("Inference error as a function of $\sigma^2$")
    plt.legend(ncol=2)
    plt.xlabel("sigma squared")
    plt.ylabel("Inference Error")
    plt.savefig("{}/error_viz_l.png".format(root))

    plt.figure(3)
    plt.clf()
    for r in r_sum:
        print(r)
        plt.scatter(r[0], r[1], c=r[2])
    print(len(r_sum))
    plt.colorbar()
    plt.xlabel('lambda')
    plt.ylabel('sigma')
    plt.savefig("{}/error_viz_mse.png".format(root))
    '''
def run_4_test(root):
    runs = []
    for i in range(1,11):
        runs.append("run{}".format(i))
    model1, model2, model3, model4 = [],[],[],[]
    for run in runs:
        dir = os.path.join(root, "k=4", run)
        results = open("{}/results.txt".format(dir))
        lines = results.readlines()
        model1.append(float(lines[4].split(": ")[1]))
        model2.append(float(lines[5].split(": ")[1]))
        model3.append(float(lines[6].split(": ")[1]))
        model4.append(float(lines[7].split(": ")[1]))
    means = []
    means.append(np.mean(model1))
    means.append(np.mean(model2))
    means.append(np.mean(model3))
    means.append(np.mean(model4))

    plt.figure(1)
    plt.clf()
    plt.title("Error for each model for k=4")
    plt.scatter([1]*10, model1)
    plt.scatter([2] * 10, model2)
    plt.scatter([3] * 10, model3)
    plt.scatter([4] * 10, model4)
    plt.plot([1, 2, 3, 4], means, label='means')
    plt.xticks([1, 2, 3, 4])
    plt.xlabel("Model #")
    plt.ylabel("Inference error")
    plt.legend()
    plt.savefig("test_mult_viz/run4_q.png")

def run_3_test(root):
    runs = []
    for i in range(1,11):
        runs.append("run{}".format(i))
    model1, model2, model3 = [],[],[]
    for run in runs:
        dir = os.path.join(root, "k=3", run)
        results = open("{}/results.txt".format(dir))
        lines = results.readlines()
        model1.append(float(lines[4].split(": ")[1]))
        model2.append(float(lines[5].split(": ")[1]))
        model3.append(float(lines[6].split(": ")[1]))
    means = []
    means.append(np.mean(model1))
    means.append(np.mean(model2))
    means.append(np.mean(model3))

    plt.figure(1)
    plt.clf()
    plt.title("Error for each model for k=3")
    plt.scatter([1]*10, model1)
    plt.scatter([2] * 10, model2)
    plt.scatter([3] * 10, model3)
    plt.plot([1, 2, 3], means, label='means')
    plt.xticks([1, 2, 3])
    plt.xlabel("Model #")
    plt.ylabel("Inference error")
    plt.legend()
    plt.savefig("test_mult_viz/run3_q.png")

if __name__ == '__main__':
    '''
    dirs = []
    for i in range(10):
        dirs.append("run_11.2/run{}".format(i))
    for dir in dirs:
        collate_results(dir)
    '''

    #read_results('summary', 'summary')

    #mult_results("test_mult")
    run_3_test("test_mult")
    run_4_test("test_mult")