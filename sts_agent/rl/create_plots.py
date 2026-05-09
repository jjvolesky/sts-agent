import matplotlib.pyplot as plt
import pandas as pd

WINDOW = 50


def plot_with_rolling(series, title, filename, color, x_label="Episode"):
    rolling = series.rolling(window=WINDOW, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(series, color=color, alpha=0.2, linewidth=0.5, label="Raw")
    ax.plot(rolling, color=color, linewidth=2, label=f"Rolling mean ({WINDOW})")
    ax.set_title(title)
    ax.set_xlabel(x_label)

    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig.savefig(filename, dpi=120)
    plt.close(fig)


def main():
    loss_df = pd.read_csv(
        "sts_agent/rl/loss_log.txt", names=["policy_loss", "value_loss"]
    )

    plot_with_rolling(
        loss_df["policy_loss"],
        "Policy Loss",
        "sts_agent/rl/policy_loss.png",
        color="blue",
    )
    plot_with_rolling(
        loss_df["value_loss"],
        "Value Loss",
        "sts_agent/rl/value_loss.png",
        color="orange",
    )

    training_df = pd.read_csv(
        "sts_agent/rl/training_log.txt", names=["act", "floor", "combats"]
    )

    plot_with_rolling(
        training_df["floor"],
        "Floor Reached",
        "sts_agent/rl/floors.png",
        color="red",
        x_label="Game",
    )
    plot_with_rolling(
        training_df["combats"],
        "Combats Survived",
        "sts_agent/rl/combats.png",
        color="green",
        x_label="Game",
    )


if __name__ == "__main__":
    main()
