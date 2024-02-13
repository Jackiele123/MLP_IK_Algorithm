import matplotlib.pyplot as plt
import pandas as pd

output_data = pd.read_csv('output_data.csv')
plt.figure(figsize=(10, 6))
plt.hist(output_data['Distance'], bins=50, edgecolor='black', alpha=0.7)
plt.title('Histogram of Losses',fontsize=20)
plt.xlabel('Loss',fontsize=16)
plt.ylabel('Frequency',fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
max_loss = output_data['Distance'].max()
min_loss = output_data['Distance'].min()
mean_loss = output_data['Distance'].mean()
quartiles = output_data['Distance'].quantile([0.25, 0.5, 0.75]).to_dict()
stats_text = f"Min: {min_loss:.2f}\nMax: {max_loss:.2f}\nMean: {mean_loss:.2f}\n"\
             f"Quartiles:\n25%: {quartiles[0.25]:.2f}\n50%: {quartiles[0.5]:.2f}\n75%: {quartiles[0.75]:.2f}"

plt.text(x=0.95, y=0.95, s=stats_text, transform=plt.gca().transAxes,
         fontsize=20, verticalalignment='top', horizontalalignment='right',
         bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))

plt.show()
# # File paths for the uploaded data (sorted by learning rates)
# file_paths = {
#     '1e-05': 'lr_epochs/weights_1e-05_losses.txt',
#     '5e-05': 'lr_epochs/weights_5e-05_losses.txt',
#     '0.0001': 'lr_epochs/weights_0.0001_losses.txt',
#     '0.0005': 'lr_epochs/weights_0.0005_losses.txt',
#     '0.00075': 'lr_epochs/weights_0.00075_losses.txt',
#     '0.001': 'lr_epochs/weights_0.001_losses.txt',
#     '0.01': 'lr_epochs/weights_0.01_losses.txt'
# }

# # Colors for each learning rate
# colors = {
#     '1e-05': 'black',
#     '5e-05': 'blue',
#     '0.0001': 'green',
#     '0.0005': 'magenta',
#     '0.00075': 'yellow',
#     '0.001': 'red',
#     '0.01': 'cyan'
# }

# # Plotting
# plt.figure(figsize=(12, 8))

# for lr, file_path in file_paths.items():
#     # Read data from text file
#     with open(file_path, 'r') as file:
#         epochs = []
#         losses = []
#         for line in file:
#             epoch, loss = line.strip().split(':')
#             epochs.append(int(epoch.split()[1]))  # Extracting the epoch number
#             losses.append(float(loss.strip()))  # Extracting the loss value

#         # Plot
#         plt.plot(epochs, losses, label=f'LR = {lr}', color=colors[lr])

# plt.title("Loss vs Epoch for Various Learning Rates", fontsize=20)
# plt.xlabel("Epoch", fontsize=16)
# plt.ylabel("Loss", fontsize=16)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend(title="Learning Rates", loc="upper right", ncol=1,fontsize=12,title_fontsize=14)
# plt.grid(True)
# plt.show()