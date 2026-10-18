"""
Plot Push-T performance across the 4 dataset size

Usage:
python pusht200_plot.py -i pusht_200/eval -o pusht_200/results.png
"""
import json
import pathlib
import click
import numpy as np
import matplotlib
matplotlib.use('Agg')   # write a file, no window needed
import matplotlib.pyplot as plt

N_DEMOS = [50, 100, 150, 200]

@click.command()
@click.option('-i', '--eval_dir', default='pusht_200/eval')
@click.option('-o', '--output', default='pusht_200/results.png')
def main(eval_dir, output):
    rewards = dict()
    #setup
    for n in N_DEMOS:
        log = json.load(open(pathlib.Path(eval_dir) / f'demos{n}' / 'eval_log.json'))
        rewards[n] = np.array([v for k, v in sorted(log.items()) 
                               if k.startswith('test/sim_max_reward_')])

    mean = np.array([rewards[n].mean() for n in N_DEMOS])
    std = np.array([rewards[n].std() for n in N_DEMOS])
    succ = np.array([(rewards[n] >= 1.0).mean() for n in N_DEMOS])
    n_ep = len(rewards[N_DEMOS[0]])
    assert all(len(rewards[n]) == n_ep for n in N_DEMOS)

    #plot
    BLUE, INK, INK2, GRID = '#2a78d6', '#0b0b0b', '#52514e', '#e4e3df'
    fig, ax1 = plt.subplots(figsize=(6, 4.2))

    rng = np.random.default_rng(0)   # fixed jitter so the figure is identical
    
    for n in N_DEMOS:
        ax1.scatter(n + rng.uniform(-6, 6, len(rewards[n])), rewards[n],
                    s=10, color=BLUE, alpha=0.2, linewidths=0)
    ax1.fill_between(N_DEMOS, mean - std, mean + std, color=BLUE, alpha=0.15, lw=0)
    ax1.plot(N_DEMOS, mean, color=BLUE, lw=2, marker='o', ms=8, mec='white', mew=2)

    for n, m, s in zip(N_DEMOS, mean, std):
        ax1.annotate(f'{m:.2f} ± {s:.2f}', (n, m), xytext=(0, -16), textcoords='offset points',
                     ha='center', color=INK, fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none'))

    ax1.set_xticks(N_DEMOS)
    ax1.set_xlim(30, 220)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlabel('Number of training demonstrations')
    ax1.set_ylabel('Test reward')
    ax1.set_title('Diffusion Policy on Push-T: performance vs. dataset size', color=INK, loc='left', fontsize=11)
    ax1.grid(axis='y', color=GRID, lw=0.8)
    ax1.set_axisbelow(True)

    for side in ('top', 'right'):
        ax1.spines[side].set_visible(False)
    fig.text(0.01, 0.01, f'Mean ± std (shaded) over the same {n_ep} test seeds for every model; '
             'dots = individual episodes.', color=INK2, fontsize=7.5)

    # save
    fig.tight_layout()
    fig.savefig(output, dpi=200, facecolor='white')
    print(f'saved {output}')
    
if __name__ == '__main__':
    main()