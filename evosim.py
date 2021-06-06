from __future__ import annotations

import argparse
import dataclasses
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--pop-size', type=int, default=100)
parser.add_argument('--generations', type=int, default=100)
parser.add_argument('--random-seed', type=int, default=0)
parser.add_argument('--generations-per-frame', type=int, default=1)

@dataclasses.dataclass(frozen=True)
class World:
  temperature: float
  pop: Population

@dataclasses.dataclass(frozen=True)
class Population:
  optimal_temps: np.ndarray
  mutation_rates: np.ndarray

  @property
  def size(self) -> int:
    return len(self.optimal_temps)

  def step(self, temperature: float) -> Population:
    fitnesses = np.exp(-(self.optimal_temps - temperature)**2)
    parent_inds = np.random.choice(np.arange(self.size), size=self.size, p=fitnesses/fitnesses.sum())
    return Population(
      optimal_temps=self.optimal_temps[parent_inds] + np.random.normal(0, np.exp(self.mutation_rates[parent_inds])),
      mutation_rates=self.mutation_rates[parent_inds] + np.random.normal(0, np.exp(self.mutation_rates[parent_inds])),
    )

@dataclasses.dataclass
class PlotBounds:
  xmin: float
  xmax: float
  ymin: float
  ymax: float

  @staticmethod
  def from_population(pop: Population) -> PlotBounds:
    return PlotBounds(
      xmin=pop.optimal_temps.min(),
      xmax=pop.optimal_temps.max(),
      ymin=pop.mutation_rates.min(),
      ymax=pop.mutation_rates.max(),
    )

  def union(self, other: PlotBounds) -> PlotBounds:
    return PlotBounds(
      xmin=min(self.xmin, other.xmin),
      xmax=max(self.xmax, other.xmax),
      ymin=min(self.ymin, other.ymin),
      ymax=max(self.ymax, other.ymax),
    )

def main(args: argparse.Namespace):
  np.random.seed(args.random_seed)

  worlds = [World(
    temperature=0,
    pop=Population(
      optimal_temps=np.random.normal(0, 1, size=args.pop_size),
      mutation_rates=np.random.normal(-3, 1, size=args.pop_size),
    ),
  )]
  for _ in range(args.generations):
    worlds.append(World(temperature=worlds[-1].temperature, pop=worlds[-1].pop.step(worlds[-1].temperature)))

  frame_worlds = worlds[::args.generations_per_frame]

  point_size = min(1, 1e3/args.pop_size)
  bounds = PlotBounds.from_population(worlds[0].pop)
  for frame, world in enumerate(frame_worlds, start=1):
    bounds = bounds.union(PlotBounds.from_population(world.pop))
    plt.clf()
    plt.scatter(world.pop.optimal_temps, world.pop.mutation_rates, s=point_size)
    plt.plot([world.temperature, world.temperature], [bounds.ymin, bounds.ymax], color='black')
    plt.xlim(bounds.xmin, bounds.xmax)
    plt.ylim(bounds.ymin, bounds.ymax)
    plt.savefig(f'frame-{frame:05d}.png')

  # point_size = min(1, 1e4/(args.pop_size * args.generations))

  # plt.clf()
  # frame_worlds = worlds[::args.generations_per_frame]
  # for frame, pop in enumerate(frame_worlds, start=1):
  #   plt.scatter(pop.optimal_temps, pop.mutation_rates, s=point_size, color=(0, 0, 0, frame/len(frame_worlds)))
  # plt.plot([true_temp, true_temp], [np.array([pop.mutation_rates for pop in worlds]).min(), np.array([pop.mutation_rates for pop in worlds]).max()], color='black')
  # plt.savefig(f'frames-overlaid.png')
  # plt.show()

if __name__ == '__main__':
  main(parser.parse_args())
