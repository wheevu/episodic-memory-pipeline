"""
Rendering utilities for the episodic memory CLI.

This module contains Rich-based rendering helpers for consistent output formatting.
"""
from typing import Any, Dict, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Global console instance
console = Console()


def format_status_icon(status: bool, warning_if_false: bool = False) -> str:
    """Return a Rich-formatted YES/NO status icon.

    Args:
        status: Boolean status value to render.
        warning_if_false: If True, render a warning style when `status` is False.

    Returns:
        A Rich markup string representing the status.
    """
    if status:
        return "[green]✓ YES[/green]"
    elif warning_if_false:
        return "[yellow]⚠ NO[/yellow]"
    else:
        return "[dim]NO[/dim]"


def format_bool_display(value: bool) -> str:
    """Display a boolean as colored YES/NO.

    Args:
        value: Boolean value to render.

    Returns:
        A Rich markup string representing the boolean.
    """
    return "[green]YES[/green]" if value else "[dim]NO[/dim]"


def format_env_value(value: Optional[str], default: str = "[dim]not set[/dim]") -> str:
    """Format an environment variable value for display.

    Args:
        value: Environment variable value, or None if not set.
        default: Default Rich markup string if `value` is None.

    Returns:
        A Rich markup string for the environment variable value.
    """
    if value is None:
        return default
    return f"[cyan]{value}[/cyan]"


def render_eval_metrics(metrics: Dict[str, Any], verbose: bool = False) -> None:
    """Render evaluation metrics in a formatted table.

    Args:
        metrics: Metrics dictionary (typically from `EvalRunResult.metrics`).
        verbose: If True, show additional details when available.

    Returns:
        None.
    """
    # Summary counts
    counts = metrics.get("counts", {})
    
    table = Table(title="Evaluation Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_column("Details")
    
    table.add_row("Episodes Ingested", str(counts.get("episodes", 0)), "")
    table.add_row("Facts Extracted", str(counts.get("facts", 0)), "")
    table.add_row("Summaries Created", str(counts.get("summaries", 0)), "")
    
    console.print(table)
    
    # Retrieval metrics
    retrieval = metrics.get("retrieval")
    if retrieval:
        table2 = Table(title="Retrieval Metrics")
        table2.add_column("Metric")
        table2.add_column("Value")
        
        precision = retrieval.get("precision_at_k", 0)
        precision_color = "green" if precision >= 0.6 else "yellow" if precision >= 0.4 else "red"
        
        table2.add_row(
            f"Precision@{retrieval.get('k', 5)}",
            f"[{precision_color}]{precision:.1%}[/{precision_color}]"
        )
        table2.add_row("Recall", f"{retrieval.get('recall', 0):.1%}")
        table2.add_row("F1 Score", f"{retrieval.get('f1', 0):.1%}")
        table2.add_row(
            "Relevant Found",
            f"{retrieval.get('relevant_found', 0)}/{retrieval.get('total_expected', 0)}"
        )
        
        console.print(table2)
    
    # Fact consistency
    fact_consistency = metrics.get("fact_consistency")
    if fact_consistency:
        table3 = Table(title="Fact Consistency Metrics")
        table3.add_column("Metric")
        table3.add_column("Value")
        
        consistency = fact_consistency.get("consistency_rate", 1.0)
        consistency_color = "green" if consistency >= 0.9 else "yellow" if consistency >= 0.7 else "red"
        
        table3.add_row(
            "Consistency Rate",
            f"[{consistency_color}]{consistency:.1%}[/{consistency_color}]"
        )
        table3.add_row("Conflict Rate", f"{fact_consistency.get('conflict_rate', 0):.1%}")
        table3.add_row(
            "Conflicting Facts",
            f"{fact_consistency.get('conflicting_facts', 0)}/{fact_consistency.get('total_facts', 0)}"
        )
        
        console.print(table3)
    
    # Compression
    compression = metrics.get("compression")
    if compression:
        table4 = Table(title="Consolidation Compression")
        table4.add_column("Metric")
        table4.add_column("Value")
        
        ratio = compression.get("ratio", 0)
        if ratio == 0:
            ratio_display = "N/A (no summaries)"
            ratio_color = "dim"
        else:
            ratio_display = f"{ratio:.2f}"
            ratio_color = "green" if 0.1 <= ratio <= 0.4 else "yellow"
        
        table4.add_row("Compression Ratio", f"[{ratio_color}]{ratio_display}[/{ratio_color}]")
        table4.add_row("Source Tokens", str(compression.get("source_tokens", 0)))
        table4.add_row("Summary Tokens", str(compression.get("summary_tokens", 0)))
        
        if ratio > 0:
            reduction = (1 - ratio) * 100
            table4.add_row("Size Reduction", f"{reduction:.0f}%")
        
        console.print(table4)


def render_eval_comparison(comparison: Any) -> None:
    """Render a comparison between two evaluation runs.

    Args:
        comparison: `EvalComparison` object returned by `EvaluationService.compare_runs`.

    Returns:
        None.
    """
    # Warnings
    for warning in comparison.warnings:
        console.print(f"[yellow]⚠ {warning}[/yellow]")
    
    console.print()
    
    # Config differences
    if comparison.config_diffs:
        table1 = Table(title="Configuration Differences")
        table1.add_column("Setting")
        table1.add_column(f"Run A ({comparison.run_a_id})")
        table1.add_column(f"Run B ({comparison.run_b_id})")
        
        for key, (val_a, val_b) in comparison.config_diffs.items():
            table1.add_row(key, str(val_a), str(val_b))
        
        console.print(table1)
    else:
        console.print("[dim]No configuration differences[/dim]")
    
    console.print()
    
    # Metric differences
    if comparison.metric_diffs:
        table2 = Table(title="Metric Differences")
        table2.add_column("Metric")
        table2.add_column(f"Run A")
        table2.add_column(f"Run B")
        table2.add_column("Delta")
        
        for key, (val_a, val_b, delta) in comparison.metric_diffs.items():
            if delta is not None:
                # Format delta with color
                if abs(delta) < 0.001:
                    delta_str = "[dim]~0[/dim]"
                elif delta > 0:
                    delta_str = f"[green]+{delta:.3f}[/green]"
                else:
                    delta_str = f"[red]{delta:.3f}[/red]"
            else:
                delta_str = "[dim]N/A[/dim]"
            
            # Format values
            if isinstance(val_a, float):
                val_a_str = f"{val_a:.3f}"
            else:
                val_a_str = str(val_a)
            
            if isinstance(val_b, float):
                val_b_str = f"{val_b:.3f}"
            else:
                val_b_str = str(val_b)
            
            table2.add_row(key, val_a_str, val_b_str, delta_str)
        
        console.print(table2)
    else:
        console.print("[dim]No metric differences[/dim]")

