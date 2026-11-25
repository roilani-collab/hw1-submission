"""Find minimum-cost campaign hubs using A* over service groups."""

from collections import deque
import csv
from dataclasses import dataclass
from pathlib import Path
import heapq
from itertools import count


Graph = dict[str, set[str]]


def load_graph(path: str | Path) -> Graph:
    """Read the headerless CSV as an undirected graph, retaining isolated nodes."""
    graph: Graph = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        for line, row in enumerate(csv.reader(source), start=1):
            if len(row) != 2 or not all(value.strip() for value in row):
                raise ValueError(f"Invalid adjacency row {line}: {row!r}")
            a, b = (value.strip() for value in row)
            graph.setdefault(a, set())
            graph.setdefault(b, set())
            if a != b:
                graph[a].add(b)
                graph[b].add(a)
    return graph


def bfs_distances_and_parents(graph: Graph, start: str):
    """Compute edge distances; absent nodes are unreachable from start."""
    if start not in graph:
        raise ValueError(f"Unknown county: {start}")
    distances = {start: 0}
    parents: dict[str, str | None] = {start: None}
    frontier = deque([start])
    while frontier:
        current = frontier.popleft()
        for neighbor in sorted(graph[current]):
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                parents[neighbor] = current
                frontier.append(neighbor)
    return distances, parents


@dataclass(frozen=True)
class SearchState:
    """R: bit mask of unserved demand atoms; k: remaining hub budget.

    g is stored separately by A*: different paths may reach this same state.
    """

    remaining: int
    hubs_left: int


def heuristic(state: SearchState) -> int:
    """h = max(0, r-k), after combining repeated (county, color) demands."""
    return max(0, state.remaining.bit_count() - state.hubs_left)


@dataclass(frozen=True)
class Demand:
    county: str
    color: str
    weight: int
    component: int


@dataclass(frozen=True)
class ServiceGroup:
    mask: int
    county: str
    color: str
    cost: int


@dataclass
class Solution:
    status: str
    hubs: list[tuple[str, str]]
    assignments: list[tuple[str, str] | None]
    paths: list[list[str] | None]
    cost: int | float
    heuristic_value: int | float


def connected_components(graph: Graph):
    component_of = {}
    components = []
    for start in sorted(graph):
        if start in component_of:
            continue
        component_id = len(components)
        members = []
        component_of[start] = component_id
        frontier = [start]
        while frontier:
            county = frontier.pop()
            members.append(county)
            for neighbor in graph[county]:
                if neighbor not in component_of:
                    component_of[neighbor] = component_id
                    frontier.append(neighbor)
        components.append(sorted(members))
    return component_of, components


def normalize_locations(locations, graph, component_of):
    """Combine repeated (county, colour) pairs, preserving input positions."""
    if not isinstance(locations, (list, tuple)) or len(locations) > 10:
        raise ValueError("locations must be a list/tuple of at most 10 pairs")
    unique = {}
    pairs = []
    weights = []
    original_indices = []
    for location in locations:
        if not isinstance(location, (tuple, list)) or len(location) != 2:
            raise ValueError("Each location must be a (county, color) pair")
        county, color = location
        if not isinstance(county, str) or county not in graph:
            raise ValueError(f"Unknown county: {county!r}")
        if color not in ("Red", "Blue"):
            raise ValueError("Party color must be Red or Blue")
        pair = (county, color)
        if pair not in unique:
            unique[pair] = len(pairs)
            pairs.append(pair)
            weights.append(0)
        atom = unique[pair]
        weights[atom] += 1
        original_indices.append(atom)
    atoms = [Demand(county, color, weights[i], component_of[county])
             for i, (county, color) in enumerate(pairs)]
    return atoms, original_indices


def bucket_masks(atoms):
    buckets = {}
    for index, atom in enumerate(atoms):
        key = (atom.component, atom.color)
        buckets[key] = buckets.get(key, 0) | (1 << index)
    return buckets


def is_feasible(state, buckets):
    return sum(bool(state.remaining & mask) for mask in buckets.values()) <= state.hubs_left


def precompute_group_costs(atoms, components, distances):
    """Find the county with the lowest weighted distance sum for each group."""
    buckets = {}
    for i, atom in enumerate(atoms):
        buckets.setdefault((atom.component, atom.color), []).append(i)
    groups = {}
    for (component, color), indices in buckets.items():
        size = 1 << len(indices)
        global_masks = [0] * size
        best_cost = [float("inf")] * size
        best_county = [None] * size
        for mask in range(1, size):
            bit = mask & -mask
            local_index = bit.bit_length() - 1
            global_masks[mask] = global_masks[mask ^ bit] | (1 << indices[local_index])
        # Sorted counties plus strict improvements break ties lexicographically.
        for county in components[component]:
            contribution = [atoms[i].weight * distances[atoms[i].county][county]
                            for i in indices]
            sums = [0] * size
            for mask in range(1, size):
                bit = mask & -mask
                sums[mask] = sums[mask ^ bit] + contribution[bit.bit_length() - 1]
                if sums[mask] < best_cost[mask]:
                    best_cost[mask] = sums[mask]
                    best_county[mask] = county
        for mask in range(1, size):
            global_mask = global_masks[mask]
            groups[global_mask] = ServiceGroup(global_mask, best_county[mask], color,
                                               best_cost[mask])
    return groups


def expand_state(state, groups, buckets):
    """All legal groups containing the first unserved atom, in mask order."""
    if not state.remaining or state.hubs_left <= 0:
        return
    anchor = state.remaining & -state.remaining
    bucket = next(mask for mask in buckets.values() if mask & anchor)
    available = state.remaining & bucket
    candidates = []
    subset = available
    while subset:
        if subset & anchor:
            candidates.append(subset)
        subset = (subset - 1) & available
    for mask in sorted(candidates):
        child = SearchState(state.remaining ^ mask, state.hubs_left - 1)
        if is_feasible(child, buckets):
            yield child, groups[mask]


def astar_search(atom_count, hub_number, groups, buckets):
    """Expand the state with smallest g + h until all requests have a hub."""
    start = SearchState((1 << atom_count) - 1, hub_number)
    sequence = count()
    start_h = heuristic(start)
    queue = [(start_h, start_h, next(sequence), 0, start)]
    best_g = {start: 0}
    parents = {}
    while queue:
        _, h, _, g, state = heapq.heappop(queue)
        if g != best_g[state]:
            continue
        if state.remaining == 0:
            chosen = []
            cursor = state
            while cursor != start:
                cursor, group = parents[cursor]
                chosen.append(group)
            return list(reversed(chosen)), g, h
        for child, group in expand_state(state, groups, buckets):
            candidate_g = g + group.cost
            if candidate_g >= best_g.get(child, float("inf")):
                continue
            best_g[child] = candidate_g
            parents[child] = (state, group)
            child_h = heuristic(child)
            heapq.heappush(queue, (candidate_g + child_h, child_h, next(sequence),
                                   candidate_g, child))
    return None, float("inf"), float("inf")


def complete_hubs(hubs, hub_number, graph, atoms):
    """Pad with distinct, possibly unused hubs; do not alter assignments."""
    present = set(hubs)
    preferred = sorted({atom.color for atom in atoms}) or ["Red"]
    colors = preferred + [c for c in ("Red", "Blue") if c not in preferred]
    for color in colors:
        for county in sorted(graph):
            if len(hubs) == hub_number:
                return hubs
            hub = (county, color)
            if hub not in present:
                hubs.append(hub)
                present.add(hub)
    if len(hubs) != hub_number:
        raise ValueError("Not enough distinct county/color pairs for the requested hubs")
    return hubs


def reconstruct_solution(chosen, atoms, original_indices, bfs_parents, hub_number, graph):
    atom_hubs = {}
    hubs = []
    for group in chosen:
        hub = (group.county, group.color)
        if hub not in hubs:
            hubs.append(hub)
        for i in range(len(atoms)):
            if group.mask & (1 << i):
                atom_hubs[i] = hub
    complete_hubs(hubs, hub_number, graph, atoms)
    assignments = []
    paths = []
    for atom_index in original_indices:
        atom = atoms[atom_index]
        hub = atom_hubs[atom_index]
        assignments.append(hub)
        path = [hub[0]]
        while path[-1] != atom.county:
            path.append(bfs_parents[atom.county][path[-1]])
        paths.append(list(reversed(path)))
    return hubs, assignments, paths


def solve_hubs(locations, hub_number, *, graph=None):
    """Return the minimum-cost assignment for the given counties."""
    if type(hub_number) is not int or not 1 <= hub_number <= 5:
        raise ValueError("hub_number must be an integer from 1 to 5")
    if graph is None:
        module_dir = Path(__file__).resolve().parent
        for folder in (module_dir, Path.cwd(), module_dir.parent):
            data = folder / "adjacency.csv"
            if data.is_file():
                graph = load_graph(data)
                break
        else:
            raise FileNotFoundError("Place the course adjacency.csv beside q1.py.")
    if hub_number > 2 * len(graph):
        raise ValueError("The map has too few distinct county/color pairs")
    component_of, components = connected_components(graph)
    atoms, original_indices = normalize_locations(locations, graph, component_of)
    buckets = bucket_masks(atoms)
    start = SearchState((1 << len(atoms)) - 1, hub_number)
    if not is_feasible(start, buckets):
        return Solution("infeasible", [], [None] * len(locations),
                        [None] * len(locations), float("inf"), float("inf"))
    distances = {}
    bfs_parents = {}
    for county in dict.fromkeys(atom.county for atom in atoms):
        distances[county], bfs_parents[county] = bfs_distances_and_parents(graph, county)
    groups = precompute_group_costs(atoms, components, distances)
    chosen, cost, goal_h = astar_search(len(atoms), hub_number, groups, buckets)
    if chosen is None:
        raise RuntimeError("Feasibility check and search disagree")
    hubs, assignments, paths = reconstruct_solution(chosen, atoms, original_indices,
                                                    bfs_parents, hub_number, graph)
    if sum(len(path) - 1 for path in paths) != cost:
        raise RuntimeError("Reconstructed path costs disagree with A*")
    return Solution("optimal", hubs, assignments, paths, cost, goal_h)


def format_output(solution, detail_output=False):
    """Match the statement's headings, order, and semicolon path delimiter."""
    def label(hub):
        return "No hub assigned" if hub is None else f"{hub[0]} ({hub[1][0]})"

    def braced(items):
        return "{" + " ; ".join(items) + "}"

    lines = ["Hubs:", braced(label(hub) for hub in solution.hubs),
             "Assignments:", braced(label(hub) for hub in solution.assignments),
             "Cost:", str(solution.cost)]
    if detail_output:
        lines += ["Heuristic:", str(solution.heuristic_value)]
    lines.append("Paths:")
    for path in solution.paths:
        if path is None:
            lines += ["No hub assigned", "Cost: inf"]
        else:
            lines += [braced(path), f"Cost: {len(path) - 1}"]
    return "\n".join(lines)


def find_hub(locations, hub_number, search_method, detail_output):
    """Print required output and return a Solution for programmatic inspection."""
    if type(search_method) is not int or search_method != 1:
        raise ValueError("Only search_method=1 (A*) is supported")
    if not isinstance(detail_output, (bool, int)) or detail_output not in (0, 1):
        raise ValueError("detail_output must be a boolean or 0/1")
    solution = solve_hubs(locations, hub_number)
    print(format_output(solution, bool(detail_output)))
    return solution
