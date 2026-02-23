"""Genetic algorithm search for representative subset selection."""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Any

import numpy as np
import pandas as pd

from .objective_driven import ObjectiveDrivenSearchAlgorithm
from .fitness import FitnessStrategy, WeightedSumFitness
from ..results import RepSetResult

if TYPE_CHECKING:
    from ..combi_gens import CombinationGenerator
    from ..context import ProblemContext
    from ..objectives import ObjectiveSet
    from ..selection_policies import SelectionPolicy
    from ..types import SliceCombination


class GeneticAlgorithmSearch(ObjectiveDrivenSearchAlgorithm):
    """Generate-and-test search using a genetic algorithm.

    Evolves a population of candidate k-combinations over multiple
    generations. Each generation applies tournament selection, crossover,
    and mutation to produce offspring, while preserving elite individuals.
    Fitness is computed by a pluggable ``FitnessStrategy``.

    This algorithm is suitable for large combination spaces where exhaustive
    enumeration is infeasible (e.g. 52-choose-8 = 752M combinations).

    Args:
        objective_set: Collection of score components defining quality metrics.
        selection_policy: Strategy for selecting the best combination from
            the final evaluated population.
        combination_generator: Defines validity constraints and k.
        fitness_strategy: Strategy for computing fitness from scores.
            Defaults to ``WeightedSumFitness()``.
        population_size: Number of individuals per generation.
        n_generations: Number of evolutionary generations.
        mutation_rate: Probability of mutating each offspring.
        crossover_rate: Probability of applying crossover (vs. cloning parent).
        elite_fraction: Fraction of top individuals copied unchanged to the
            next generation.
        tournament_size: Number of candidates in tournament selection.
        seed: Random seed for reproducibility.

    Examples:

        >>> from energy_repset import ObjectiveSet, WeightedSumPolicy
        >>> from energy_repset.score_components import WassersteinFidelity
        >>> from energy_repset.combi_gens import ExhaustiveCombiGen
        >>> from energy_repset.search_algorithms import GeneticAlgorithmSearch
        >>> objectives = ObjectiveSet({"wass": (1.0, WassersteinFidelity())})
        >>> algo = GeneticAlgorithmSearch(
        ...     objective_set=objectives,
        ...     selection_policy=WeightedSumPolicy(),
        ...     combination_generator=ExhaustiveCombiGen(k=4),
        ...     population_size=50,
        ...     n_generations=100,
        ...     seed=42,
        ... )
    """

    def __init__(
        self,
        objective_set: ObjectiveSet,
        selection_policy: SelectionPolicy,
        combination_generator: CombinationGenerator,
        fitness_strategy: Optional[FitnessStrategy] = None,
        population_size: int = 50,
        n_generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elite_fraction: float = 0.1,
        tournament_size: int = 3,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(objective_set, selection_policy)
        self.combination_generator = combination_generator
        self.fitness_strategy = fitness_strategy or WeightedSumFitness()
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_fraction = elite_fraction
        self.tournament_size = tournament_size
        self.seed = seed

    def find_selection(self, context: ProblemContext) -> RepSetResult:
        """Evolve a population to find the best selection.

        Args:
            context: Problem context with ``df_features`` populated.

        Returns:
            RepSetResult with the best selection, scores, representatives,
            and diagnostics containing ``evaluations_df`` (final generation)
            and ``generation_history`` (per-generation stats).
        """
        from tqdm import tqdm

        self.objective_set.prepare(context)
        unique_slices = context.get_unique_slices()
        k = self.combination_generator.k
        rng = np.random.default_rng(self.seed)

        population = self._initialize_population(unique_slices, k, rng)
        generation_history: List[Dict[str, Any]] = []

        for gen in tqdm(range(self.n_generations), desc="GA generations"):
            evaluations_df = self._evaluate_population(population, context)
            fitness = self.fitness_strategy.rank(evaluations_df, self.objective_set)

            generation_history.append({
                "generation": gen,
                "best_fitness": float(np.max(fitness)),
                "mean_fitness": float(np.mean(fitness)),
                "std_fitness": float(np.std(fitness)),
            })

            population = self._evolve(
                population, fitness, unique_slices, k, rng
            )

        final_evaluations_df = self._evaluate_population(population, context)
        winning_combination = self.selection_policy.select_best(
            final_evaluations_df, self.objective_set
        )

        slice_labels = context.slicer.labels_for_index(context.df_raw.index)
        return RepSetResult(
            context=context,
            selection_space="subset",
            selection=winning_combination,
            scores=self.objective_set.evaluate(winning_combination, context),
            representatives={
                s: context.df_raw.iloc[slice_labels == s] for s in winning_combination
            },
            diagnostics={
                "evaluations_df": final_evaluations_df,
                "generation_history": pd.DataFrame(generation_history),
            },
        )

    def _initialize_population(
        self,
        unique_slices: list,
        k: int,
        rng: np.random.Generator,
    ) -> List[SliceCombination]:
        """Create an initial population of random valid individuals.

        Args:
            unique_slices: Available slice labels.
            k: Combination size.
            rng: Random generator.

        Returns:
            List of valid SliceCombination tuples.
        """
        population: List[SliceCombination] = []
        max_attempts = self.population_size * 20

        attempts = 0
        while len(population) < self.population_size and attempts < max_attempts:
            attempts += 1
            indices = rng.choice(len(unique_slices), size=k, replace=False)
            combi = tuple(sorted(unique_slices[i] for i in indices))
            if self.combination_generator.combination_is_valid(combi, unique_slices):
                population.append(combi)

        return population

    def _evaluate_population(
        self,
        population: List[SliceCombination],
        context: ProblemContext,
    ) -> pd.DataFrame:
        """Score every individual in the population.

        Args:
            population: List of candidate combinations.
            context: Problem context for evaluation.

        Returns:
            DataFrame with columns ``slices``, ``label``, and one column
            per score component.
        """
        rows = []
        for combi in population:
            metrics = self.objective_set.evaluate(combi, context)
            rec = {
                "slices": combi,
                "label": ", ".join(str(s) for s in combi),
            }
            rec.update(metrics)
            rows.append(rec)
        return pd.DataFrame(rows)

    def _evolve(
        self,
        population: List[SliceCombination],
        fitness: np.ndarray,
        unique_slices: list,
        k: int,
        rng: np.random.Generator,
    ) -> List[SliceCombination]:
        """Produce the next generation via elitism, selection, crossover, mutation.

        Args:
            population: Current generation.
            fitness: Fitness values (higher = better).
            unique_slices: Available slice labels.
            k: Combination size.
            rng: Random generator.

        Returns:
            New population of the same size.
        """
        pop_size = len(population)
        n_elite = max(1, int(pop_size * self.elite_fraction))

        elite_indices = np.argsort(fitness)[-n_elite:]
        new_population = [population[i] for i in elite_indices]

        while len(new_population) < pop_size:
            parent1 = self._tournament_select(population, fitness, rng)
            if rng.random() < self.crossover_rate:
                parent2 = self._tournament_select(population, fitness, rng)
                child = self._crossover(parent1, parent2, k, unique_slices, rng)
            else:
                child = parent1

            if rng.random() < self.mutation_rate:
                child = self._mutate(child, unique_slices, rng)

            new_population.append(child)

        return new_population

    def _tournament_select(
        self,
        population: List[SliceCombination],
        fitness: np.ndarray,
        rng: np.random.Generator,
    ) -> SliceCombination:
        """Select one individual via tournament selection.

        Args:
            population: Current population.
            fitness: Fitness values.
            rng: Random generator.

        Returns:
            The fittest individual among the tournament contestants.
        """
        indices = rng.choice(len(population), size=self.tournament_size, replace=False)
        best = indices[np.argmax(fitness[indices])]
        return population[best]

    def _crossover(
        self,
        parent1: SliceCombination,
        parent2: SliceCombination,
        k: int,
        unique_slices: list,
        rng: np.random.Generator,
    ) -> SliceCombination:
        """Pool-based crossover: draw k elements from the union of both parents.

        Retries up to 20 times if the offspring is invalid; falls back to
        parent1 on failure.

        Args:
            parent1: First parent combination.
            parent2: Second parent combination.
            k: Combination size.
            unique_slices: Available slice labels.
            rng: Random generator.

        Returns:
            Valid offspring combination.
        """
        gene_pool = list(set(parent1) | set(parent2))
        for _ in range(20):
            if len(gene_pool) < k:
                break
            chosen = rng.choice(len(gene_pool), size=k, replace=False)
            child = tuple(sorted(gene_pool[i] for i in chosen))
            if self.combination_generator.combination_is_valid(child, unique_slices):
                return child
        return parent1

    def _mutate(
        self,
        individual: SliceCombination,
        unique_slices: list,
        rng: np.random.Generator,
    ) -> SliceCombination:
        """Swap one gene for a random non-selected slice.

        Retries up to 20 times if the mutant is invalid; falls back to the
        original individual on failure.

        Args:
            individual: Combination to mutate.
            unique_slices: Available slice labels.
            rng: Random generator.

        Returns:
            Mutated (or original) combination.
        """
        current = set(individual)
        available = [s for s in unique_slices if s not in current]
        if not available:
            return individual

        for _ in range(20):
            drop_idx = rng.integers(len(individual))
            new_gene = available[rng.integers(len(available))]
            genes = list(individual)
            genes[drop_idx] = new_gene
            mutant = tuple(sorted(genes))
            if self.combination_generator.combination_is_valid(mutant, unique_slices):
                return mutant
        return individual
