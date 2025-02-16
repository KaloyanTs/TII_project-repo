import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.ticker import MultipleLocator

csv_file = "training_log.csv"
data = pd.read_csv(csv_file)

required_columns = ["iteration", "loss", "best_perplexity"]
if not all(column in data.columns for column in required_columns):
    raise ValueError(f"CSV file must contain the following columns: {required_columns}")


plt.figure(figsize=(10, 5))
plt.plot(data["iteration"], data["loss"], label="Крос-ентропия", color="blue")
plt.xlabel("Итерации")
plt.ylabel("Крос-ентропия")
plt.grid(True)
y_ticks = np.arange(data["loss"].min(), data["loss"].max() + 0.2, 0.2)
plt.yticks(y_ticks)

max_iteration = data["iteration"].max()
x_ticks = [tick for tick in plt.xticks()[0] if tick >= 0 and tick<max_iteration]
if max_iteration not in x_ticks:
    x_ticks.append(max_iteration)
x_ticks = sorted(x_ticks)
plt.xticks(x_ticks, fontsize=7)

plt.gca().yaxis.set_minor_locator(MultipleLocator(0.1))
plt.minorticks_on()

plt.savefig("loss.svg", format="svg", bbox_inches="tight", pad_inches=0)
plt.savefig("loss.png", format="png", bbox_inches="tight", pad_inches=0)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(data["iteration"], data["best_perplexity"], label="Перплексия", color="green")
plt.xlabel("Итерации")
plt.ylabel("Перплексия")

plt.grid(True)
y_ticks = np.arange(math.ceil(data["best_perplexity"].min()), min(data["best_perplexity"].max(), 10) + 0.1, 0.5)
y_ticks = np.insert(y_ticks, 0, data["best_perplexity"].min())
plt.yticks(y_ticks)

plt.xticks(x_ticks, fontsize=7)

plt.gca().yaxis.set_minor_locator(MultipleLocator(0.1))
plt.minorticks_on()

plt.ylim(bottom=4, top=11)

plt.savefig("perplexity.svg", format="svg", bbox_inches="tight", pad_inches=0)
plt.savefig("perplexity.png", format="png", bbox_inches="tight", pad_inches=0)
plt.show()