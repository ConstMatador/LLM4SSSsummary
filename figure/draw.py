import matplotlib.pyplot as plt

selected_model = "GPT4SSS"

# File Path
file_path = f"figure/error/{selected_model}.txt"
save_path = f"figure/figure/{selected_model}.png"

data = []
try:
    with open(file_path, "r") as file:
        for line in file:
            try:
                value = float(line.strip())
                data.append(value)
            except ValueError:
                print(f"{line.strip()} can not be transformed to float")
except FileNotFoundError:
    print(f"{file_path} not found")

if len(data) != 100:
    print(f"Error: The number of data points is not 100, it is{len(data)}")

plt.plot(data)
plt.title(selected_model)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.xticks(range(1, 101,5))

plt.savefig(save_path)