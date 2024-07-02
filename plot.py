import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def main(argv):
    work_name = str(argv[1])
    file = os.path.join(work_name, "loss_log.txt")
    savepath = os.path.dirname(work_name)
    G_GAN = []
    G_GAN_Feat = []
    G_VGG = []
    D_real = []
    D_fake = []
    epoch = []

    with open(file,'r') as f:
        lines = f.readlines()
        for line in lines:
            if "(epoch" not in line: continue
            epoch.append(int(line.split()[1].split(",")[0]))
            if "G_GAN" in line :
                G_GAN_list = line.split()
                G_GAN.append(float(G_GAN_list[7]))
            if "G_GAN_Feat" in line :
                G_GAN_Feat_list = line.split()
                G_GAN_Feat.append(float(G_GAN_Feat_list[9]))
            if "G_VGG" in line :
                G_VGG_list = line.split()
                G_VGG.append(float(G_VGG_list[11]))
            if "D_real" in line:
                D_real_list = line.split()
                D_real.append(float(D_real_list[13]))
            if "D_fake" in line:
                D_fake_list = line.split()
                D_fake.append(float(D_fake_list[15]))
    
    del_ind = []
    for i in range(len(epoch)-1):
        if epoch[i] == epoch[i+1]:
            del_ind.append(i)
        else: pass

    for i in sorted(del_ind, reverse=True):
        del G_GAN[i]
        del G_GAN_Feat[i]
        del G_VGG[i]
        del D_real[i]
        del D_fake[i]

    plt.figure()#(figsize=(50,6))
    ax = np.linspace(0,len(G_GAN)-1,len(G_GAN))
    plt.plot(ax,G_GAN,label="G_GAN")
    plt.plot(ax,G_GAN_Feat,label="G_GAN_Feat")
    plt.plot(ax,G_VGG,label="G_VGG")
    plt.plot(ax,D_real,label="D_real")
    plt.plot(ax,D_fake,label="D_fake")
    
    plt.grid(color='gray', linestyle='--', linewidth=1, alpha=0.3)
    plt.legend()
    plt.title('pix2pixHD$_1$')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(savepath, os.path.splitext(os.path.basename(file))[0]+".png"))
    plt.show()

if __name__ == "__main__":
    main(sys.argv)