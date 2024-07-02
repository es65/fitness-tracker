import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
path = '../../data/interim/01_data_processed.parquet'
df = pd.read_parquet(path)

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------

set_df = df[df['set'] == 1]

plt.plot(set_df['acc_y'])
plt.plot(set_df['acc_y'].reset_index(drop=True))

# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------

labels = df['label'].unique()

for label in labels:
    subset = df[df['label'] == label]
    # display(subset.head(2))
    fig, ax = plt.subplots()
    plt.plot(subset['acc_y'].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()

for label in labels:
    subset = df[df['label'] == label]
    # display(subset.head(2))
    fig, ax = plt.subplots()
    plt.plot(subset[:100]['acc_y'].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()
# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------

mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100

# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------
cat_df = df.query("label == 'ohp'").query("participant == 'B'").reset_index()

fig, ax = plt.subplots()
cat_df.groupby(['category'])['acc_y'].plot()
ax.set_ylabel('acc_y')
ax.set_xlabel('samples')
plt.legend()


# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------

fig, ax = plt.subplots()
participant_df = df.query("label == 'bench'").sort_values('participant').reset_index()
participant_df.groupby(['participant'])['acc_y'].plot()
ax.set_ylabel('acc_y')
ax.set_xlabel('samples')
plt.legend()

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------

label = 'squat'
participant = 'A'
all_axes_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index()

fig, ax = plt.subplots()
all_axes_df[['acc_x', 'acc_y', 'acc_z']].plot(ax=ax)
ax.set_ylabel('acc_y')
ax.set_xlabel('samples')
plt.legend()

# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------

labels = df['label'].unique()
participants = sorted(df['participant'].unique())

for label in labels:
    for participant in participants:
        all_axes_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )
        if len(all_axes_df) > 0:
            fig, ax = plt.subplots()
            all_axes_df[['acc_x', 'acc_y', 'acc_z']].plot(ax=ax)
            ax.set_xlabel('samples')
            plt.title(f"{label} ({participant})".title())
            plt.legend()

for label in labels:
    for participant in participants:
        all_axes_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )
        if len(all_axes_df) > 0:
            fig, ax = plt.subplots()
            all_axes_df[['gyr_x', 'gyr_y', 'gyr_z']].plot(ax=ax)
            ax.set_xlabel('samples')
            plt.title(f"{label} ({participant})".title())
            plt.legend()

# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------

label = 'squat'
participant = 'A'
combined_plot_df = (
    df.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index(drop=True)
    )

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(20,10))
combined_plot_df[['acc_x', 'acc_y', 'acc_z']].plot(ax=ax[0])
combined_plot_df[['gyr_x', 'gyr_y', 'gyr_z']].plot(ax=ax[1])

ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncols=3, fancybox=True, shadow=True)
ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncols=3, fancybox=True, shadow=True)
ax[1].set_xlabel('samples')

# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------

labels = df['label'].unique()
participants = sorted(df['participant'].unique())

for label in labels:
    for participant in participants:
        combined_plot_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )
        
        if len(combined_plot_df) > 0:
            fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(20,10))
            combined_plot_df[['acc_x', 'acc_y', 'acc_z']].plot(ax=ax[0])
            combined_plot_df[['gyr_x', 'gyr_y', 'gyr_z']].plot(ax=ax[1])

            ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncols=3, fancybox=True, shadow=True)
            ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncols=3, fancybox=True, shadow=True)
            ax[1].set_xlabel('samples')
            
            fig.tight_layout(pad=3.0)
            fig.suptitle(f"{label.title()} - Participant {participant}", fontsize=16, ha='left', x=0.02)
            
            plt.savefig(f"../../reports/figures/{label}_{participant}.png")
            plt.show()