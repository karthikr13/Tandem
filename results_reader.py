import os

def read_results(root):
    r_sum = []
    for subdirs, _, files in os.walk(root):
        for file in files:
            if file != 'results.txt':
                continue
            results = open(os.path.join(subdirs, file))
            lines = results.readlines()
            lines = [line.rstrip() for line in lines]
            sig = lines[2].split(": ")[1]
            lam = lines[3].split(": ")[1]
            inf_err1 = lines[4].split("be ")[1]
            inf_err2= lines[5].split("be ")[1]
            inf_err = lines[6].split("be ")[1]
            r = (sig, lam, inf_err, inf_err1, inf_err2)
            r_sum.append(r)
            results.close()
    r_sum.sort(key=lambda r: (r[0], r[1]))
    write_file = open(os.path.join(root, "summary.csv"), "w")
    write_file.write("sig squared, lambda, inference error, model 1 error, model 2 error\n")
    for result in r_sum:
        for item in result:
            write_file.write(item)
            write_file.write(",")
        write_file.write("\n")
    write_file.close()
if __name__ == '__main__':
    read_results("runs-10.27")