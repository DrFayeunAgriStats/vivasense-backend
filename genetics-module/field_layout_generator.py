"""VivaSense field layout generator engine.

This module generates deterministic field layouts for classic and Pro designs.
It returns both:
- plot_matrix: nested structures for visual rendering
- fieldbook: flat records for Excel export

Pro access checks are intentionally not handled here. Route-level code should
read DESIGN_REGISTRY[design]["requires_pro"] and enforce entitlement.
"""

from __future__ import annotations

from itertools import product
import math
from pprint import pprint
import random
from typing import Any, Callable, Dict, List, Sequence, Tuple, TypeVar


T = TypeVar("T")
PlotMatrix = List[List[Dict[str, Any]]]
Fieldbook = List[Dict[str, Any]]
GeneratorResult = Dict[str, Any]
ValidatorFn = Callable[[Dict[str, Any]], None]
GeneratorFn = Callable[[Dict[str, Any], random.Random], GeneratorResult]


_GLOBAL_RANDOM = random.Random()


def set_seed(seed: int) -> None:
    """Set the module-level random seed.

    Parameters
    ----------
    seed : int
        Seed value used to initialize deterministic randomization.

    Returns
    -------
    None
    """
    _GLOBAL_RANDOM.seed(seed)


def shuffle_copy(items: List[T], rng: random.Random) -> List[T]:
    """Return a shuffled copy of a list without mutating the input.

    Parameters
    ----------
    items : list[T]
        Items to shuffle.
    rng : random.Random
        Random number generator instance.

    Returns
    -------
    list[T]
        New shuffled list.
    """
    copied = list(items)
    rng.shuffle(copied)
    return copied


def _require_list_of_labels(value: Any, field_name: str) -> List[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"'{field_name}' must be a non-empty list of treatment labels.")
    labels = [str(v).strip() for v in value]
    if any(not label for label in labels):
        raise ValueError(f"'{field_name}' contains empty labels. Provide valid treatment names.")
    if len(set(labels)) != len(labels):
        raise ValueError(f"'{field_name}' contains duplicate labels. Labels must be unique.")
    return labels


def _require_positive_int(value: Any, field_name: str, minimum: int = 1) -> int:
    if not isinstance(value, int):
        raise ValueError(f"'{field_name}' must be an integer.")
    if value < minimum:
        raise ValueError(f"'{field_name}' must be >= {minimum}.")
    return value


def _require_common_numeric_fields(request: Dict[str, Any]) -> None:
    for key in ("plot_width_m", "plot_length_m"):
        val = request.get(key)
        if val is None:
            continue
        if not isinstance(val, (int, float)) or float(val) <= 0:
            raise ValueError(f"'{key}' must be a positive number.")

    aisle = request.get("aisle_width_m")
    if aisle is not None and (not isinstance(aisle, (int, float)) or float(aisle) < 0):
        raise ValueError("'aisle_width_m' must be zero or a positive number.")


def _build_color_index(labels: Sequence[str]) -> Dict[str, int]:
    return {label: idx for idx, label in enumerate(labels)}


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    for divisor in range(2, int(math.sqrt(value)) + 1):
        if value % divisor == 0:
            return False
    return True


def _shape_summary(request: Dict[str, Any], title: str, n_plots: int, n_treatments: int) -> Dict[str, Any]:
    return {
        "title": title,
        "n_plots": n_plots,
        "n_treatments": n_treatments,
        "replications": request.get("replications"),
        "plot_width_m": request.get("plot_width_m"),
        "plot_length_m": request.get("plot_length_m"),
        "aisle_width_m": request.get("aisle_width_m"),
        "seed": request.get("seed"),
    }


def _verify_latin_square_properties(plot_matrix: PlotMatrix, treatments: Sequence[str]) -> None:
    """Verify each treatment appears exactly once per row and once per column.

    Parameters
    ----------
    plot_matrix : list[list[dict]]
        Latin-square matrix output.
    treatments : Sequence[str]
        Full treatment labels.

    Returns
    -------
    None
    """
    expected = sorted(treatments)
    n = len(plot_matrix)

    for row_idx, row in enumerate(plot_matrix, start=1):
        row_treatments = sorted(cell["treatment"] for cell in row)
        if row_treatments != expected:
            raise ValueError(
                f"Latin square row {row_idx} is invalid. "
                "Each treatment must appear exactly once per row."
            )

    for col_idx in range(n):
        col_treatments = sorted(plot_matrix[row_idx][col_idx]["treatment"] for row_idx in range(n))
        if col_treatments != expected:
            raise ValueError(
                f"Latin square column {col_idx + 1} is invalid. "
                "Each treatment must appear exactly once per column."
            )


def _assert_fieldbook_columns(fieldbook: Fieldbook, expected_columns: Sequence[str], design_type: str) -> None:
    """Validate fieldbook key order and exact schema for a design.

    Parameters
    ----------
    fieldbook : list[dict]
        Fieldbook rows.
    expected_columns : Sequence[str]
        Ordered expected key names.
    design_type : str
        Design label used in error messages.

    Returns
    -------
    None
    """
    expected = list(expected_columns)
    for row_idx, row in enumerate(fieldbook, start=1):
        row_columns = list(row.keys())
        if row_columns != expected:
            raise ValueError(
                f"Fieldbook schema mismatch for '{design_type}' at row {row_idx}. "
                f"Expected columns {expected}, got {row_columns}."
            )


def _verify_balanced_lattice_pairwise(plot_matrix: PlotMatrix, treatments: Sequence[str]) -> None:
    """Verify every treatment pair appears in the same block exactly once.

    Parameters
    ----------
    plot_matrix : list[list[dict]]
        Balanced-lattice plot matrix.
    treatments : Sequence[str]
        Treatment labels.

    Returns
    -------
    None
    """
    from collections import defaultdict

    pair_counts = defaultdict(int)
    all_pairs = {
        tuple(sorted((treatments[i], treatments[j])))
        for i in range(len(treatments))
        for j in range(i + 1, len(treatments))
    }

    for rep_blocks in plot_matrix:
        for block in rep_blocks:
            block_treatments = [plot["treatment"] for plot in block["plots"]]
            for i, t1 in enumerate(block_treatments):
                for t2 in block_treatments[i + 1 :]:
                    pair = tuple(sorted((t1, t2)))
                    pair_counts[pair] += 1

    violations = [(pair, pair_counts.get(pair, 0)) for pair in sorted(all_pairs) if pair_counts.get(pair, 0) != 1]
    if violations:
        raise ValueError(
            f"Balanced lattice pairwise balance violated for {len(violations)} pairs. "
            "Every pair must appear in the same block exactly once."
        )


def _verify_alpha_lattice_structure(
    plot_matrix: PlotMatrix,
    treatments: Sequence[str],
    block_size: int,
) -> None:
    """Verify alpha-lattice block structure and treatment coverage.

    Parameters
    ----------
    plot_matrix : list[list[dict]]
        Alpha-lattice plot matrix.
    treatments : Sequence[str]
        Treatment labels.
    block_size : int
        Block size used in the design.

    Returns
    -------
    None
    """
    expected_blocks = len(treatments) // block_size
    expected_treatments = sorted(treatments)

    for rep_idx, rep_blocks in enumerate(plot_matrix, start=1):
        if len(rep_blocks) != expected_blocks:
            raise ValueError(
                f"Alpha lattice replication {rep_idx} has {len(rep_blocks)} blocks; "
                f"expected {expected_blocks}."
            )

        rep_treatments: List[str] = []
        for block in rep_blocks:
            block_treatments = [plot["treatment"] for plot in block["plots"]]
            if len(block_treatments) != block_size:
                raise ValueError(
                    f"Alpha lattice block '{block['block_id']}' has {len(block_treatments)} treatments; "
                    f"expected {block_size}."
                )
            rep_treatments.extend(block_treatments)

        if sorted(rep_treatments) != expected_treatments:
            raise ValueError(
                f"Alpha lattice replication {rep_idx} does not contain each treatment exactly once."
            )


def validate_crd(request: Dict[str, Any]) -> None:
    """Validate CRD request payload.

    Parameters
    ----------
    request : dict
        Field layout request.

    Returns
    -------
    None
    """
    _require_list_of_labels(request.get("treatments"), "treatments")
    _require_positive_int(request.get("replications"), "replications", minimum=1)
    _require_common_numeric_fields(request)


def generate_crd(request: Dict[str, Any], rng: random.Random) -> GeneratorResult:
    """Generate a CRD layout.

    Parameters
    ----------
    request : dict
        Field layout request.
    rng : random.Random
        Deterministic RNG.

    Returns
    -------
    dict
        Layout payload containing plot_matrix, fieldbook, and summary.
    """
    treatments = _require_list_of_labels(request.get("treatments"), "treatments")
    reps = _require_positive_int(request.get("replications"), "replications", minimum=1)
    color_index = _build_color_index(treatments)

    pool: List[str] = []
    for _rep in range(reps):
        pool.extend(treatments)

    shuffled = shuffle_copy(pool, rng)
    total_plots = len(shuffled)

    requested_cols = request.get("columns")
    if isinstance(requested_cols, int) and requested_cols > 0:
        n_cols = requested_cols
    else:
        n_cols = math.ceil(math.sqrt(total_plots))

    requested_rows = request.get("rows")
    if isinstance(requested_rows, int) and requested_rows > 0:
        n_rows = requested_rows
    else:
        n_rows = math.ceil(total_plots / n_cols)

    plot_matrix: PlotMatrix = []
    fieldbook: Fieldbook = []
    rep_counter: Dict[str, int] = {t: 0 for t in treatments}

    plot_id = 1
    idx = 0
    for row in range(1, n_rows + 1):
        row_cells: List[Dict[str, Any]] = []
        for col in range(1, n_cols + 1):
            if idx >= total_plots:
                continue
            treatment = shuffled[idx]
            rep_counter[treatment] += 1
            row_cells.append(
                {
                    "plot_id": plot_id,
                    "treatment": treatment,
                    "color_index": color_index[treatment],
                }
            )
            fieldbook.append(
                {
                    "plot_id": plot_id,
                    "rep": rep_counter[treatment],
                    "row": row,
                    "column": col,
                    "treatment": treatment,
                }
            )
            plot_id += 1
            idx += 1
        if row_cells:
            plot_matrix.append(row_cells)

    return {
        "plot_matrix": plot_matrix,
        "fieldbook": fieldbook,
        "layout_summary": _shape_summary(request, "Completely Randomized Design (CRD)", total_plots, len(treatments)),
        "alpha_value": None,
    }


def validate_rcbd(request: Dict[str, Any]) -> None:
    """Validate RCBD request payload.

    Parameters
    ----------
    request : dict
        Field layout request.

    Returns
    -------
    None
    """
    _require_list_of_labels(request.get("treatments"), "treatments")
    _require_positive_int(request.get("replications"), "replications", minimum=2)
    _require_common_numeric_fields(request)


def generate_rcbd(request: Dict[str, Any], rng: random.Random) -> GeneratorResult:
    """Generate an RCBD layout.

    Parameters
    ----------
    request : dict
        Field layout request.
    rng : random.Random
        Deterministic RNG.

    Returns
    -------
    dict
        Layout payload containing plot_matrix, fieldbook, and summary.
    """
    treatments = _require_list_of_labels(request.get("treatments"), "treatments")
    reps = _require_positive_int(request.get("replications"), "replications", minimum=2)
    color_index = _build_color_index(treatments)

    n_cols = request.get("columns") if isinstance(request.get("columns"), int) and request.get("columns") > 0 else len(treatments)

    plot_matrix: PlotMatrix = []
    fieldbook: Fieldbook = []
    plot_id = 1

    for rep in range(1, reps + 1):
        order = shuffle_copy(treatments, rng)
        rep_row: List[Dict[str, Any]] = []
        for idx, treatment in enumerate(order):
            rep_row.append(
                {
                    "plot_id": plot_id,
                    "rep": rep,
                    "treatment": treatment,
                    "color_index": color_index[treatment],
                }
            )
            fieldbook.append(
                {
                    "plot_id": plot_id,
                    "rep": rep,
                    "block": rep,
                    "row": rep,
                    "column": (idx % n_cols) + 1,
                    "treatment": treatment,
                }
            )
            plot_id += 1
        plot_matrix.append(rep_row)

    return {
        "plot_matrix": plot_matrix,
        "fieldbook": fieldbook,
        "layout_summary": _shape_summary(
            request,
            "Randomized Complete Block Design (RCBD)",
            len(treatments) * reps,
            len(treatments),
        ),
        "alpha_value": None,
    }


def validate_latin_square(request: Dict[str, Any]) -> None:
    """Validate Latin square request.

    Parameters
    ----------
    request : dict
        Field layout request.

    Returns
    -------
    None
    """
    treatments = _require_list_of_labels(request.get("treatments"), "treatments")
    rows = _require_positive_int(request.get("rows"), "rows", minimum=4)
    cols = _require_positive_int(request.get("columns"), "columns", minimum=4)

    if rows != cols:
        raise ValueError("Latin square requires equal numbers of rows and columns (n x n).")
    n = len(treatments)
    if n < 4 or n > 10:
        raise ValueError("Latin square supports 4 to 10 treatments (n = 4..10).")
    if n != rows:
        raise ValueError(
            "Latin square requires n treatments = n rows = n columns. "
            f"Received treatments={n}, rows={rows}, columns={cols}."
        )
    _require_common_numeric_fields(request)


def generate_latin_square(request: Dict[str, Any], rng: random.Random) -> GeneratorResult:
    """Generate an n x n Latin square using cyclic construction and row/column shuffles.

    Parameters
    ----------
    request : dict
        Field layout request.
    rng : random.Random
        Deterministic RNG.

    Returns
    -------
    dict
        Layout payload containing plot_matrix, fieldbook, and summary.
    """
    treatments = _require_list_of_labels(request.get("treatments"), "treatments")
    n = len(treatments)
    color_index = _build_color_index(treatments)

    base_square = [[treatments[(row + col) % n] for col in range(n)] for row in range(n)]
    row_order = shuffle_copy(list(range(n)), rng)
    col_order = shuffle_copy(list(range(n)), rng)

    plot_matrix: PlotMatrix = []
    fieldbook: Fieldbook = []
    plot_id = 1

    for out_row, src_row in enumerate(row_order, start=1):
        matrix_row: List[Dict[str, Any]] = []
        for out_col, src_col in enumerate(col_order, start=1):
            treatment = base_square[src_row][src_col]
            cell = {
                "plot_id": plot_id,
                "treatment": treatment,
                "color_index": color_index[treatment],
            }
            matrix_row.append(cell)
            fieldbook.append(
                {
                    "plot_id": plot_id,
                    "row": out_row,
                    "column": out_col,
                    "treatment": treatment,
                }
            )
            plot_id += 1
        plot_matrix.append(matrix_row)

    _verify_latin_square_properties(plot_matrix, treatments)
    _assert_fieldbook_columns(
        fieldbook,
        ["plot_id", "row", "column", "treatment"],
        "latin_square",
    )

    return {
        "plot_matrix": plot_matrix,
        "fieldbook": fieldbook,
        "layout_summary": _shape_summary(request, "Latin Square Design", n * n, n),
        "alpha_value": None,
    }


def validate_split_plot(request: Dict[str, Any]) -> None:
    """Validate split-plot request.

    Parameters
    ----------
    request : dict
        Field layout request.

    Returns
    -------
    None
    """
    _require_list_of_labels(request.get("main_treatments"), "main_treatments")
    _require_list_of_labels(request.get("sub_treatments"), "sub_treatments")
    _require_positive_int(request.get("replications"), "replications", minimum=2)
    _require_common_numeric_fields(request)


def generate_split_plot(request: Dict[str, Any], rng: random.Random) -> GeneratorResult:
    """Generate split-plot design with independent randomization at each stratum.

    Parameters
    ----------
    request : dict
        Field layout request.
    rng : random.Random
        Deterministic RNG.

    Returns
    -------
    dict
        Layout payload containing plot_matrix, fieldbook, and summary.
    """
    main_treatments = _require_list_of_labels(request.get("main_treatments"), "main_treatments")
    sub_treatments = _require_list_of_labels(request.get("sub_treatments"), "sub_treatments")
    reps = _require_positive_int(request.get("replications"), "replications", minimum=2)

    color_index = _build_color_index(sub_treatments)
    plot_matrix: PlotMatrix = []
    fieldbook: Fieldbook = []

    plot_id = 1
    main_plot_uid = 1

    for rep in range(1, reps + 1):
        rep_row: List[Dict[str, Any]] = []
        randomized_main = shuffle_copy(main_treatments, rng)
        for main_idx, main_treatment in enumerate(randomized_main, start=1):
            randomized_sub = shuffle_copy(sub_treatments, rng)
            sub_nodes: List[Dict[str, Any]] = []

            for sub_idx, sub_treatment in enumerate(randomized_sub, start=1):
                sub_nodes.append(
                    {
                        "sub_plot_id": sub_idx,
                        "plot_id": plot_id,
                        "sub_treatment": sub_treatment,
                        "color_index": color_index[sub_treatment],
                    }
                )
                fieldbook.append(
                    {
                        "plot_id": plot_id,
                        "rep": rep,
                        "main_plot": main_idx,
                        "main_treatment": main_treatment,
                        "sub_plot": sub_idx,
                        "sub_treatment": sub_treatment,
                    }
                )
                plot_id += 1

            rep_row.append(
                {
                    "plot_id": main_plot_uid,
                    "rep": rep,
                    "main_plot": main_idx,
                    "main_treatment": main_treatment,
                    "sub_plots": sub_nodes,
                }
            )
            main_plot_uid += 1
        plot_matrix.append(rep_row)

    total_plots = reps * len(main_treatments) * len(sub_treatments)

    _assert_fieldbook_columns(
        fieldbook,
        [
            "plot_id",
            "rep",
            "main_plot",
            "main_treatment",
            "sub_plot",
            "sub_treatment",
        ],
        "split_plot",
    )

    return {
        "plot_matrix": plot_matrix,
        "fieldbook": fieldbook,
        "layout_summary": _shape_summary(request, "Split-Plot Design", total_plots, len(main_treatments) * len(sub_treatments)),
        "alpha_value": None,
    }


def validate_split_split_plot(request: Dict[str, Any]) -> None:
    """Validate split-split plot request.

    Parameters
    ----------
    request : dict
        Field layout request.

    Returns
    -------
    None
    """
    _require_list_of_labels(request.get("main_treatments"), "main_treatments")
    _require_list_of_labels(request.get("sub_treatments"), "sub_treatments")
    _require_list_of_labels(request.get("sub_sub_treatments"), "sub_sub_treatments")
    _require_positive_int(request.get("replications"), "replications", minimum=2)
    _require_common_numeric_fields(request)


def generate_split_split_plot(request: Dict[str, Any], rng: random.Random) -> GeneratorResult:
    """Generate split-split plot with three nested randomization levels.

    Parameters
    ----------
    request : dict
        Field layout request.
    rng : random.Random
        Deterministic RNG.

    Returns
    -------
    dict
        Layout payload containing plot_matrix, fieldbook, and summary.
    """
    main_treatments = _require_list_of_labels(request.get("main_treatments"), "main_treatments")
    sub_treatments = _require_list_of_labels(request.get("sub_treatments"), "sub_treatments")
    sub_sub_treatments = _require_list_of_labels(request.get("sub_sub_treatments"), "sub_sub_treatments")
    reps = _require_positive_int(request.get("replications"), "replications", minimum=2)

    color_index = _build_color_index(sub_sub_treatments)
    plot_matrix: PlotMatrix = []
    fieldbook: Fieldbook = []

    plot_id = 1
    main_plot_uid = 1

    for rep in range(1, reps + 1):
        rep_row: List[Dict[str, Any]] = []
        randomized_main = shuffle_copy(main_treatments, rng)

        for main_idx, main_treatment in enumerate(randomized_main, start=1):
            randomized_sub = shuffle_copy(sub_treatments, rng)
            sub_nodes: List[Dict[str, Any]] = []

            for sub_idx, sub_treatment in enumerate(randomized_sub, start=1):
                randomized_sub_sub = shuffle_copy(sub_sub_treatments, rng)
                sub_sub_nodes: List[Dict[str, Any]] = []

                for sub_sub_idx, sub_sub_treatment in enumerate(randomized_sub_sub, start=1):
                    sub_sub_nodes.append(
                        {
                            "sub_sub_plot_id": sub_sub_idx,
                            "plot_id": plot_id,
                            "sub_sub_treatment": sub_sub_treatment,
                            "color_index": color_index[sub_sub_treatment],
                        }
                    )
                    fieldbook.append(
                        {
                            "plot_id": plot_id,
                            "rep": rep,
                            "main_plot": main_idx,
                            "main_treatment": main_treatment,
                            "sub_plot": sub_idx,
                            "sub_treatment": sub_treatment,
                            "sub_sub_plot": sub_sub_idx,
                            "sub_sub_treatment": sub_sub_treatment,
                        }
                    )
                    plot_id += 1

                sub_nodes.append(
                    {
                        "sub_plot_id": sub_idx,
                        "sub_treatment": sub_treatment,
                        "sub_sub_plots": sub_sub_nodes,
                    }
                )

            rep_row.append(
                {
                    "plot_id": main_plot_uid,
                    "main_plot": main_idx,
                    "main_treatment": main_treatment,
                    "sub_plots": sub_nodes,
                }
            )
            main_plot_uid += 1

        plot_matrix.append(rep_row)

    total_plots = reps * len(main_treatments) * len(sub_treatments) * len(sub_sub_treatments)

    _assert_fieldbook_columns(
        fieldbook,
        [
            "plot_id",
            "rep",
            "main_plot",
            "main_treatment",
            "sub_plot",
            "sub_treatment",
            "sub_sub_plot",
            "sub_sub_treatment",
        ],
        "split_split",
    )

    return {
        "plot_matrix": plot_matrix,
        "fieldbook": fieldbook,
        "layout_summary": _shape_summary(
            request,
            "Split-Split Plot Design",
            total_plots,
            len(main_treatments) * len(sub_treatments) * len(sub_sub_treatments),
        ),
        "alpha_value": None,
    }


def _factor_code(index: int) -> str:
    if index < 26:
        return chr(ord("A") + index)
    return f"F{index + 1}"


def validate_factorial_rcbd(request: Dict[str, Any]) -> None:
    """Validate factorial RCBD request.

    Parameters
    ----------
    request : dict
        Field layout request.

    Returns
    -------
    None
    """
    factors = request.get("factors")
    if not isinstance(factors, dict) or not factors:
        raise ValueError("'factors' must be a non-empty dictionary of factor levels.")

    for factor_name, levels in factors.items():
        if not isinstance(factor_name, str) or not factor_name.strip():
            raise ValueError("Each factor must have a non-empty name.")
        _require_list_of_labels(levels, f"factors.{factor_name}")

    _require_positive_int(request.get("replications"), "replications", minimum=2)
    _require_common_numeric_fields(request)


def generate_factorial_rcbd(request: Dict[str, Any], rng: random.Random) -> GeneratorResult:
    """Generate factorial RCBD with all combinations per block.

    Parameters
    ----------
    request : dict
        Field layout request.
    rng : random.Random
        Deterministic RNG.

    Returns
    -------
    dict
        Layout payload containing plot_matrix, fieldbook, and summary.
    """
    factors_raw = request.get("factors")
    if not isinstance(factors_raw, dict):
        raise ValueError("'factors' must be provided for factorial RCBD.")

    factor_names = list(factors_raw.keys())
    factor_levels: List[List[str]] = [_require_list_of_labels(factors_raw[name], f"factors.{name}") for name in factor_names]
    reps = _require_positive_int(request.get("replications"), "replications", minimum=2)

    combinations = list(product(*factor_levels))
    color_index = {combo: idx for idx, combo in enumerate(combinations)}

    combo_objects: List[Dict[str, Any]] = []
    for combo in combinations:
        label_parts: List[str] = []
        levels_by_name: Dict[str, str] = {}
        coded_levels: Dict[str, str] = {}

        for idx, level in enumerate(combo):
            code = _factor_code(idx)
            levels = factor_levels[idx]
            level_number = levels.index(level) + 1
            label_parts.append(f"{code}{level_number}")
            levels_by_name[factor_names[idx]] = level
            coded_levels[f"factor_{code.lower()}_level"] = level

        combo_objects.append(
            {
                "combination": combo,
                "treatment_combination": "".join(label_parts),
                "factor_levels": levels_by_name,
                "coded_levels": coded_levels,
            }
        )

    plot_matrix: PlotMatrix = []
    fieldbook: Fieldbook = []
    plot_id = 1

    for rep in range(1, reps + 1):
        randomized = shuffle_copy(combo_objects, rng)
        rep_row: List[Dict[str, Any]] = []
        for combo_obj in randomized:
            rep_row.append(
                {
                    "plot_id": plot_id,
                    "treatment_combination": combo_obj["treatment_combination"],
                    "factor_levels": combo_obj["factor_levels"],
                    "color_index": color_index[combo_obj["combination"]],
                }
            )

            fb_row: Dict[str, Any] = {
                "plot_id": plot_id,
                "rep": rep,
                "block": rep,
                "treatment_combination": combo_obj["treatment_combination"],
            }
            fb_row.update(combo_obj["coded_levels"])

            fieldbook.append(fb_row)
            plot_id += 1
        plot_matrix.append(rep_row)

    n_combinations = len(combinations)
    expected_factor_columns = [f"factor_{_factor_code(i).lower()}_level" for i in range(len(factor_names))]
    _assert_fieldbook_columns(
        fieldbook,
        ["plot_id", "rep", "block", "treatment_combination", *expected_factor_columns],
        "factorial_rcbd",
    )

    return {
        "plot_matrix": plot_matrix,
        "fieldbook": fieldbook,
        "layout_summary": _shape_summary(
            request,
            "Factorial RCBD",
            n_combinations * reps,
            n_combinations,
        ),
        "alpha_value": None,
    }


def validate_balanced_lattice(request: Dict[str, Any]) -> None:
    """Validate balanced lattice request.

    Parameters
    ----------
    request : dict
        Field layout request.

    Returns
    -------
    None
    """
    treatments = _require_list_of_labels(request.get("treatments"), "treatments")
    reps = _require_positive_int(request.get("replications"), "replications", minimum=2)

    t = len(treatments)
    block_size = int(math.sqrt(t))
    if block_size * block_size != t:
        raise ValueError(
            "Balanced lattice requires a perfect-square number of treatments. "
            f"Received {t} treatments."
        )

    required_reps = block_size + 1
    if reps != required_reps:
        raise ValueError(
            "Balanced lattice requires replications = block_size + 1. "
            f"With {t} treatments, block_size={block_size}, so replications must be {required_reps}."
        )

    if not _is_prime(block_size):
        raise ValueError(
            "This balanced lattice constructor requires a prime block size so a true pairwise-balanced "
            f"affine-plane layout can be formed. Received block_size={block_size}."
        )
    _require_common_numeric_fields(request)


def generate_balanced_lattice(request: Dict[str, Any], rng: random.Random) -> GeneratorResult:
    """Generate balanced lattice layout.

    Parameters
    ----------
    request : dict
        Field layout request.
    rng : random.Random
        Deterministic RNG.

    Returns
    -------
    dict
        Layout payload containing plot_matrix, fieldbook, and summary.
    """
    treatments = _require_list_of_labels(request.get("treatments"), "treatments")
    reps = _require_positive_int(request.get("replications"), "replications", minimum=2)

    t = len(treatments)
    k = int(math.sqrt(t))
    color_index = _build_color_index(treatments)

    if not _is_prime(k):
        raise ValueError(
            "True balanced lattice construction is implemented here for prime block sizes only. "
            f"Received block_size={k}."
        )

    treatment_lookup = {
        (x, y): treatments[x * k + y]
        for x in range(k)
        for y in range(k)
    }

    rep_arrangements: List[List[List[str]]] = []

    # Rep 1: vertical lines x = c.
    rep_arrangements.append(
        [
            [treatment_lookup[(x, y)] for y in range(k)]
            for x in range(k)
        ]
    )

    # Reps 2..k+1: slope classes y = m*x + b (mod k), one replication per slope.
    for slope in range(k):
        blocks_for_slope: List[List[str]] = []
        for intercept in range(k):
            block = [
                treatment_lookup[(x, (slope * x + intercept) % k)]
                for x in range(k)
            ]
            blocks_for_slope.append(block)
        rep_arrangements.append(blocks_for_slope)

    rep_arrangements = rep_arrangements[:reps]

    plot_matrix: PlotMatrix = []
    fieldbook: Fieldbook = []
    plot_id = 1

    for rep, rep_blocks_source in enumerate(rep_arrangements, start=1):
        randomized_blocks = shuffle_copy(list(range(len(rep_blocks_source))), rng)
        rep_blocks: List[Dict[str, Any]] = []

        for block_idx, src_block in enumerate(randomized_blocks, start=1):
            block_treatments = shuffle_copy(rep_blocks_source[src_block], rng)

            plots: List[Dict[str, Any]] = []
            for treatment in block_treatments:
                plots.append(
                    {
                        "plot_id": plot_id,
                        "treatment": treatment,
                        "color_index": color_index[treatment],
                    }
                )
                fieldbook.append(
                    {
                        "plot_id": plot_id,
                        "rep": rep,
                        "block": block_idx,
                        "treatment": treatment,
                    }
                )
                plot_id += 1

            rep_blocks.append(
                {
                    "block_id": f"R{rep}B{block_idx}",
                    "plots": plots,
                }
            )
        plot_matrix.append(rep_blocks)

    _verify_balanced_lattice_pairwise(plot_matrix, treatments)

    _assert_fieldbook_columns(
        fieldbook,
        ["plot_id", "rep", "block", "treatment"],
        "lattice",
    )

    return {
        "plot_matrix": plot_matrix,
        "fieldbook": fieldbook,
        "layout_summary": _shape_summary(request, "Balanced Lattice Design", t * reps, t),
        "alpha_value": None,
    }


def validate_alpha_lattice(request: Dict[str, Any]) -> None:
    """Validate alpha lattice request.

    Parameters
    ----------
    request : dict
        Field layout request.

    Returns
    -------
    None
    """
    treatments = _require_list_of_labels(request.get("treatments"), "treatments")
    block_size = _require_positive_int(request.get("block_size"), "block_size", minimum=2)
    reps = _require_positive_int(request.get("replications"), "replications", minimum=2)

    t = len(treatments)
    if t % block_size != 0:
        raise ValueError(
            "Alpha lattice requires number of treatments divisible by block size. "
            f"Received treatments={t}, block_size={block_size}."
        )
    _require_common_numeric_fields(request)


def generate_alpha_lattice(request: Dict[str, Any], rng: random.Random) -> GeneratorResult:
    """Generate alpha lattice blocks by random shuffling within each replication.

    Parameters
    ----------
    request : dict
        Field layout request.
    rng : random.Random
        Deterministic RNG.

    Returns
    -------
    dict
        Layout payload containing plot_matrix, fieldbook, summary, and alpha value.
    """
    treatments = _require_list_of_labels(request.get("treatments"), "treatments")
    block_size = _require_positive_int(request.get("block_size"), "block_size", minimum=2)
    reps = _require_positive_int(request.get("replications"), "replications", minimum=2)

    t = len(treatments)
    n_blocks = t // block_size
    color_index = _build_color_index(treatments)

    plot_matrix: PlotMatrix = []
    fieldbook: Fieldbook = []
    plot_id = 1

    for rep in range(1, reps + 1):
        shuffled = shuffle_copy(treatments, rng)
        blocks: List[Dict[str, Any]] = []

        for block in range(1, n_blocks + 1):
            start = (block - 1) * block_size
            end = start + block_size
            block_treatments = shuffled[start:end]

            plots: List[Dict[str, Any]] = []
            for treatment in block_treatments:
                plots.append(
                    {
                        "plot_id": plot_id,
                        "treatment": treatment,
                        "color_index": color_index[treatment],
                    }
                )
                fieldbook.append(
                    {
                        "plot_id": plot_id,
                        "rep": rep,
                        "block": block,
                        "treatment": treatment,
                        "block_size": block_size,
                    }
                )
                plot_id += 1

            blocks.append(
                {
                    "block_id": f"R{rep}B{block}",
                    "plots": plots,
                }
            )

        plot_matrix.append(blocks)

    alpha_value = (reps * block_size * (block_size - 1)) / (t * (t - 1))
    _verify_alpha_lattice_structure(plot_matrix, treatments, block_size)

    _assert_fieldbook_columns(
        fieldbook,
        ["plot_id", "rep", "block", "treatment", "block_size"],
        "alpha_lattice",
    )

    summary = _shape_summary(request, "Alpha Lattice Design", t * reps, t)
    summary["block_size"] = block_size
    summary["n_blocks_per_rep"] = n_blocks
    summary["alpha_value"] = alpha_value

    return {
        "plot_matrix": plot_matrix,
        "fieldbook": fieldbook,
        "layout_summary": summary,
        "alpha_value": alpha_value,
    }


DESIGN_REGISTRY: Dict[str, Dict[str, Any]] = {
    "latin_square": {
        "validator": validate_latin_square,
        "generator": generate_latin_square,
        "requires_pro": True,
    },
    "split_plot": {
        "validator": validate_split_plot,
        "generator": generate_split_plot,
        "requires_pro": True,
    },
    "split_split": {
        "validator": validate_split_split_plot,
        "generator": generate_split_split_plot,
        "requires_pro": True,
    },
    "factorial_rcbd": {
        "validator": validate_factorial_rcbd,
        "generator": generate_factorial_rcbd,
        "requires_pro": True,
    },
    "lattice": {
        "validator": validate_balanced_lattice,
        "generator": generate_balanced_lattice,
        "requires_pro": True,
    },
    "alpha_lattice": {
        "validator": validate_alpha_lattice,
        "generator": generate_alpha_lattice,
        "requires_pro": True,
    },
    "crd": {
        "validator": validate_crd,
        "generator": generate_crd,
        "requires_pro": False,
    },
    "rcbd": {
        "validator": validate_rcbd,
        "generator": generate_rcbd,
        "requires_pro": False,
    },
}


def generate_field_layout(request: Dict[str, Any]) -> Dict[str, Any]:
    """Generate field layout for supported designs.

    Parameters
    ----------
    request : dict
        Input payload with design_type and design-specific parameters.

    Returns
    -------
    dict
        {
          "design_type": str,
          "plot_matrix": list[list[dict]],
          "fieldbook": list[dict],
          "layout_summary": dict,
          "alpha_value": float | None,
        }
    """
    design_type = str(request.get("design_type", "")).strip().lower()
    if not design_type:
        raise ValueError("'design_type' is required.")

    config = DESIGN_REGISTRY.get(design_type)
    if config is None:
        supported = ", ".join(sorted(DESIGN_REGISTRY.keys()))
        raise ValueError(f"Unsupported design_type '{design_type}'. Supported designs: {supported}.")

    seed_value = request.get("seed", 0)
    if not isinstance(seed_value, int):
        raise ValueError("'seed' must be an integer for reproducible randomization.")

    set_seed(seed_value)
    rng = random.Random(seed_value)

    validator: ValidatorFn = config["validator"]
    generator: GeneratorFn = config["generator"]

    validator(request)
    generated = generator(request, rng)

    return {
        "design_type": design_type,
        "plot_matrix": generated["plot_matrix"],
        "fieldbook": generated["fieldbook"],
        "layout_summary": generated["layout_summary"],
        "alpha_value": generated.get("alpha_value"),
    }


if __name__ == "__main__":
    def _assert_split_plot_rep_randomization(
        result: Dict[str, Any],
        main_treatments: Sequence[str],
    ) -> None:
        """Check split-plot main treatment randomization by replication.

        Parameters
        ----------
        result : dict
            Generated split-plot result.
        main_treatments : Sequence[str]
            Expected main treatments.

        Returns
        -------
        None
        """
        rep_orders: Dict[int, List[str]] = {}
        for rep_row in result["plot_matrix"]:
            if not rep_row:
                continue
            first_plot = rep_row[0]
            rep = first_plot.get("rep")
            if not isinstance(rep, int):
                raise ValueError("Split-plot plot_matrix is missing replication metadata.")
            rep_orders[rep] = [node["main_treatment"] for node in rep_row]

        expected = sorted(main_treatments)
        for rep, order in rep_orders.items():
            if sorted(order) != expected:
                raise ValueError(
                    f"Split-plot main plot randomization invalid in replication {rep}. "
                    "Each replication must contain all main treatments exactly once."
                )

    def _assert_alpha_formula(result: Dict[str, Any], request: Dict[str, Any]) -> None:
        """Validate alpha-lattice alpha value against the design formula.

        Parameters
        ----------
        result : dict
            Generated alpha-lattice result.
        request : dict
            Alpha-lattice request payload.

        Returns
        -------
        None
        """
        r = int(request["replications"])
        k = int(request["block_size"])
        t = len(request["treatments"])
        expected_alpha = (r * k * (k - 1)) / (t * (t - 1))
        actual_alpha = result.get("alpha_value")
        if actual_alpha is None or not math.isclose(actual_alpha, expected_alpha, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                "Alpha value verification failed. "
                f"Expected {expected_alpha}, got {actual_alpha}."
            )

    latin_request = {
        "design_type": "latin_square",
        "treatments": ["T1", "T2", "T3", "T4"],
        "rows": 4,
        "columns": 4,
        "plot_width_m": 2.0,
        "plot_length_m": 3.0,
        "aisle_width_m": 0.5,
        "seed": 42,
    }

    split_plot_request = {
        "design_type": "split_plot",
        "main_treatments": ["M1", "M2", "M3"],
        "sub_treatments": ["S1", "S2"],
        "replications": 3,
        "plot_width_m": 2.0,
        "plot_length_m": 3.0,
        "aisle_width_m": 0.5,
        "seed": 19,
    }

    split_split_request = {
        "design_type": "split_split",
        "main_treatments": ["M1", "M2"],
        "sub_treatments": ["S1", "S2"],
        "sub_sub_treatments": ["SS1", "SS2"],
        "replications": 2,
        "plot_width_m": 2.0,
        "plot_length_m": 3.0,
        "aisle_width_m": 0.5,
        "seed": 42,
    }

    factorial_request = {
        "design_type": "factorial_rcbd",
        "factors": {
            "Nitrogen": ["N0", "N1", "N2"],
            "Variety": ["V1", "V2", "V3"],
        },
        "replications": 3,
        "plot_width_m": 2.0,
        "plot_length_m": 3.0,
        "aisle_width_m": 0.5,
        "seed": 42,
    }

    lattice_request = {
        "design_type": "lattice",
        "treatments": [f"T{i}" for i in range(1, 10)],
        "replications": 4,
        "plot_width_m": 2.0,
        "plot_length_m": 3.0,
        "aisle_width_m": 0.5,
        "seed": 42,
    }

    alpha_request = {
        "design_type": "alpha_lattice",
        "treatments": [f"G{i}" for i in range(1, 17)],
        "block_size": 4,
        "replications": 3,
        "plot_width_m": 2.0,
        "plot_length_m": 3.0,
        "aisle_width_m": 0.5,
        "seed": 42,
    }

    test_requests: List[Tuple[str, Dict[str, Any]]] = [
        ("latin_square", latin_request),
        ("split_plot", split_plot_request),
        ("split_split", split_split_request),
        ("factorial_rcbd", factorial_request),
        ("lattice", lattice_request),
        ("alpha_lattice", alpha_request),
    ]

    print("=== VivaSense Field Layout Self-Test (6 Pro Designs) ===")
    for design_name, payload in test_requests:
        try:
            result = generate_field_layout(payload)
            if design_name == "latin_square":
                _verify_latin_square_properties(result["plot_matrix"], payload["treatments"])
            if design_name == "split_plot":
                _assert_split_plot_rep_randomization(result, payload["main_treatments"])
            if design_name == "alpha_lattice":
                _assert_alpha_formula(result, payload)

            print(f"\n[{design_name}] OK — no ValueError")
            pprint(result)
        except ValueError as exc:
            print(f"\n[{design_name}] FAILED with ValueError: {exc}")
            raise

    print("\nSelf-test completed successfully for all 6 Pro designs.")
