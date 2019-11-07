import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot():
    """
    for a in ["SGT", "hate", "offensive"]:
        losses = pd.read_csv("plots/" + a + ".csv")
        x = np.linspace(0, losses.shape[0], losses.shape[0])
        plt.figure()
        plt.plot(x, losses["test"], color = "red")
        plt.plot(x, losses["train"], color = "blue")
        plt.savefig("plots/" + a)
    """
    #plt.figure()
    losses = pd.read_csv("losses.csv")
    x = np.linspace(0, losses.shape[0], losses.shape[0])
    #plt.plot(x, losses["discriminator"], color="red")
    plt.plot(x, losses["generator"], color="blue")
    plt.plot(x, losses["recogniztion"], color="purple")
    plt.savefig("plots/losses")


if __name__ == "__main__":
    plot()