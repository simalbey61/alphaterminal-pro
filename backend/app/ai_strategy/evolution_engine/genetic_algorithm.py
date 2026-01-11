"""
AlphaTerminal Pro - Genetic Algorithm
=====================================

Strateji parametrelerinin genetik optimizasyonu.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any, Callable, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import random
from uuid import uuid4

import numpy as np

from app.ai_strategy.constants import EvolutionDefaults

logger = logging.getLogger(__name__)


@dataclass
class Gene:
    """Tek bir gen (parametre)."""
    name: str
    value: float
    min_value: float
    max_value: float
    step: Optional[float] = None
    is_integer: bool = False
    
    def mutate(self, sigma: float = 0.1) -> "Gene":
        """Gen mutasyonu."""
        range_size = self.max_value - self.min_value
        mutation = np.random.normal(0, sigma * range_size)
        new_value = self.value + mutation
        new_value = max(self.min_value, min(self.max_value, new_value))
        
        if self.step:
            new_value = round(new_value / self.step) * self.step
        if self.is_integer:
            new_value = int(round(new_value))
        
        return Gene(
            name=self.name, value=new_value, min_value=self.min_value,
            max_value=self.max_value, step=self.step, is_integer=self.is_integer,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "value": self.value, "min_value": self.min_value, "max_value": self.max_value}


@dataclass
class Chromosome:
    """Kromozom (strateji parametreleri seti)."""
    id: str
    genes: List[Gene]
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "genes": {g.name: g.value for g in self.genes},
            "fitness": self.fitness,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
        }
    
    def get_param(self, name: str) -> Optional[float]:
        for gene in self.genes:
            if gene.name == name:
                return gene.value
        return None
    
    @classmethod
    def create_random(cls, gene_specs: List[Dict], generation: int = 0) -> "Chromosome":
        genes = []
        for spec in gene_specs:
            if spec.get("is_integer"):
                value = random.randint(int(spec["min"]), int(spec["max"]))
            else:
                value = random.uniform(spec["min"], spec["max"])
                if spec.get("step"):
                    value = round(value / spec["step"]) * spec["step"]
            
            genes.append(Gene(
                name=spec["name"], value=value, min_value=spec["min"],
                max_value=spec["max"], step=spec.get("step"), is_integer=spec.get("is_integer", False),
            ))
        return cls(id=str(uuid4())[:8], genes=genes, generation=generation)


@dataclass
class EvolutionResult:
    """Evrim sonucu."""
    best_chromosome: Chromosome
    best_fitness: float
    generations_completed: int
    total_evaluations: int
    fitness_history: List[float]
    diversity_history: List[float]
    converged: bool
    convergence_generation: Optional[int]
    elapsed_time_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_chromosome": self.best_chromosome.to_dict(),
            "best_fitness": self.best_fitness,
            "generations_completed": self.generations_completed,
            "total_evaluations": self.total_evaluations,
            "converged": self.converged,
            "elapsed_time_seconds": self.elapsed_time_seconds,
        }


class GeneticAlgorithm:
    """Strateji parametreleri için genetik algoritma."""
    
    def __init__(
        self,
        population_size: int = 100,
        generations: int = 50,
        elite_pct: float = 0.10,
        crossover_prob: float = 0.50,
        mutation_prob: float = 0.10,
        mutation_sigma: float = 0.10,
        tournament_size: int = 3,
        fitness_func: Optional[Callable[[Dict[str, float]], float]] = None,
        convergence_threshold: float = 0.001,
        convergence_generations: int = 10,
    ):
        self.population_size = population_size
        self.generations = generations
        self.elite_pct = elite_pct
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.mutation_sigma = mutation_sigma
        self.tournament_size = tournament_size
        self.fitness_func = fitness_func
        self.convergence_threshold = convergence_threshold
        self.convergence_generations = convergence_generations
    
    def evolve(
        self,
        gene_specs: List[Dict],
        initial_population: Optional[List[Chromosome]] = None,
        verbose: bool = True,
    ) -> EvolutionResult:
        """Genetik algoritma çalıştır."""
        import time
        start_time = time.time()
        
        # Initialize population
        if initial_population:
            population = initial_population[:self.population_size]
            while len(population) < self.population_size:
                population.append(Chromosome.create_random(gene_specs, 0))
        else:
            population = [Chromosome.create_random(gene_specs, 0) for _ in range(self.population_size)]
        
        self._evaluate_population(population)
        
        fitness_history, diversity_history = [], []
        total_evaluations = self.population_size
        converged, convergence_gen = False, None
        best_fitness_streak = []
        
        for gen in range(self.generations):
            population.sort(key=lambda c: c.fitness, reverse=True)
            
            best_fitness = population[0].fitness
            diversity = self._calculate_diversity(population)
            
            fitness_history.append(best_fitness)
            diversity_history.append(diversity)
            
            if verbose and gen % 10 == 0:
                logger.info(f"Gen {gen}: Best={best_fitness:.4f}, Diversity={diversity:.4f}")
            
            # Convergence check
            best_fitness_streak.append(best_fitness)
            if len(best_fitness_streak) > self.convergence_generations:
                best_fitness_streak.pop(0)
                if max(best_fitness_streak) - min(best_fitness_streak) < self.convergence_threshold:
                    converged, convergence_gen = True, gen
                    break
            
            # Create next generation
            new_population = []
            elite_count = int(self.population_size * self.elite_pct)
            new_population.extend(population[:elite_count])
            
            while len(new_population) < self.population_size:
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)
                
                if random.random() < self.crossover_prob:
                    child1, child2 = self._crossover(parent1, parent2, gen + 1)
                else:
                    child1 = self._clone(parent1, gen + 1)
                    child2 = self._clone(parent2, gen + 1)
                
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            self._evaluate_population(new_population[elite_count:])
            total_evaluations += len(new_population) - elite_count
            population = new_population
        
        population.sort(key=lambda c: c.fitness, reverse=True)
        
        return EvolutionResult(
            best_chromosome=population[0],
            best_fitness=population[0].fitness,
            generations_completed=len(fitness_history),
            total_evaluations=total_evaluations,
            fitness_history=fitness_history,
            diversity_history=diversity_history,
            converged=converged,
            convergence_generation=convergence_gen,
            elapsed_time_seconds=time.time() - start_time,
        )
    
    def _evaluate_population(self, population: List[Chromosome]) -> None:
        if not self.fitness_func:
            for chrom in population:
                chrom.fitness = random.random()
            return
        
        for chrom in population:
            params = {g.name: g.value for g in chrom.genes}
            try:
                chrom.fitness = self.fitness_func(params)
            except Exception as e:
                logger.warning(f"Fitness error: {e}")
                chrom.fitness = -1000
    
    def _tournament_select(self, population: List[Chromosome]) -> Chromosome:
        contestants = random.sample(population, min(self.tournament_size, len(population)))
        return max(contestants, key=lambda c: c.fitness)
    
    def _crossover(self, parent1: Chromosome, parent2: Chromosome, generation: int) -> Tuple[Chromosome, Chromosome]:
        """Uniform crossover."""
        genes1, genes2 = [], []
        
        for g1, g2 in zip(parent1.genes, parent2.genes):
            if random.random() < 0.5:
                genes1.append(Gene(**g1.__dict__))
                genes2.append(Gene(**g2.__dict__))
            else:
                genes1.append(Gene(**g2.__dict__))
                genes2.append(Gene(**g1.__dict__))
        
        child1 = Chromosome(id=str(uuid4())[:8], genes=genes1, generation=generation, parent_ids=[parent1.id, parent2.id])
        child2 = Chromosome(id=str(uuid4())[:8], genes=genes2, generation=generation, parent_ids=[parent1.id, parent2.id])
        
        return child1, child2
    
    def _clone(self, parent: Chromosome, generation: int) -> Chromosome:
        return Chromosome(
            id=str(uuid4())[:8],
            genes=[Gene(**g.__dict__) for g in parent.genes],
            generation=generation,
            parent_ids=[parent.id],
        )
    
    def _mutate(self, chromosome: Chromosome) -> Chromosome:
        new_genes = []
        for gene in chromosome.genes:
            if random.random() < self.mutation_prob:
                new_genes.append(gene.mutate(self.mutation_sigma))
            else:
                new_genes.append(Gene(**gene.__dict__))
        
        chromosome.genes = new_genes
        return chromosome
    
    def _calculate_diversity(self, population: List[Chromosome]) -> float:
        if len(population) < 2:
            return 0.0
        
        all_values = []
        for chrom in population:
            values = [g.value for g in chrom.genes]
            all_values.append(values)
        
        arr = np.array(all_values)
        std_per_gene = np.std(arr, axis=0)
        range_per_gene = np.array([g.max_value - g.min_value for g in population[0].genes])
        
        normalized_std = std_per_gene / (range_per_gene + 1e-8)
        return float(np.mean(normalized_std))
