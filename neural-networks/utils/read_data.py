def read_mnist(file_name):
    data_set = []
    with open(file_name, "rt") as f:
        for line in f:
            line = line.replace("\n", "")
            tokens = line.split(",")
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i + 1])
            data_set.append([label, attribs])
    return data_set


def show_mnist(file_name, mode):
    data_set = read_mnist(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == "pixels":
                if data_set[obs][1][idx] == "0":
                    print(" ", end="")
                else:
                    print("*", end="")
            else:
                print("%4s " % data_set[obs][1][idx], end="")
            if (idx % 28) == 27:
                print(" ")
        print("LABEL: %s" % data_set[obs][0], end="")
        print(" ")


def read_insurability(file_name):
    count = 0
    data = []
    with open(file_name, "rt") as f:
        for line in f:
            if count > 0:
                line = line.replace("\n", "")
                tokens = line.split(",")
                if len(line) > 10:
                    x1 = float(tokens[0])
                    x2 = float(tokens[1])
                    x3 = float(tokens[2])
                    if tokens[3] == "Good":
                        cls = 0
                    elif tokens[3] == "Neutral":
                        cls = 1
                    else:
                        cls = 2
                    data.append([[cls], [x1, x2, x3]])
            count = count + 1
    return data
