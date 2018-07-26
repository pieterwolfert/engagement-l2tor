import matplotlib.pyplot as plt
from scipy.misc import imread

def main():
    source_dir = "/home/pieter/projects/engagement-l2tor/data/emotions/"
    smiling = []
    with open('./data/emotions/x_train3.txt', 'r') as f:
        x = f.readlines()
        x = [y.strip() for y in x]
        wdw = plt.figure()
        ax = wdw.gca()
        wdw.show()
        for item in x:
            ax.imshow(imread(source_dir + item))
            wdw.canvas.draw()
            sml = input("1 for smiling, 0 if not")
            smiling.append(sml)
            #show image
            #annotate smiling


if __name__ == '__main__':
    main()
