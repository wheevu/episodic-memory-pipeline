"""Evaluation commands for the episodic memory CLI."""

from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.panel import Panel
from rich.table import Table

from src.services import EvaluationService

from ..render import console, render_eval_comparison, render_eval_metrics

if TYPE_CHECKING:
    from src.bootstrap import PipelineComponents


def _get_components(ctx: click.Context) -> "PipelineComponents":
    """Get pipeline components from the Click context.

    Args:
        ctx: Click context with a `use_mock` flag stored in `ctx.obj`.

    Returns:
        A `PipelineComponents` instance created by `src.cli.get_pipeline_components`.
    """
    from src.cli import get_pipeline_components

    return get_pipeline_components(ctx.obj.get("use_mock", False))


@click.command("eval")
@click.option("--scenario", "-s", default="diary", help="Evaluation scenario to run")
@click.option("--k", default=5, help="K value for precision@k metric")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.pass_context
def eval_cmd(ctx: click.Context, scenario: str, k: int, verbose: bool) -> None:
    """Run evaluation metrics on the memory pipeline (legacy command).

    Args:
        ctx: Click context with pipeline initialization settings.
        scenario: Evaluation scenario identifier to run.
        k: K value for precision@k metric.
        verbose: If True, show more detailed output.

    Returns:
        None.
    """
    # Delegate to eval_run for backwards compatibility
    ctx.invoke(eval_run, scenario=scenario, k=k, verbose=verbose, save=False)


@click.command("eval-run")
@click.option("--scenario", "-s", default="diary", help="Evaluation scenario to run")
@click.option("--k", default=5, help="K value for precision@k metric")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.option("--save/--no-save", default=True, help="Save run to runs/eval/")
@click.option("--out", type=click.Path(), help="Custom output directory")
@click.pass_context
def eval_run(
    ctx: click.Context, scenario: str, k: int, verbose: bool, save: bool, out: str
) -> None:
    """Run a versioned evaluation and optionally save results.

    Args:
        ctx: Click context with pipeline initialization settings.
        scenario: Evaluation scenario identifier to run.
        k: K value for precision@k metric.
        verbose: If True, show more detailed output.
        save: If True, persist results to `runs/eval/`.
        out: Optional custom output directory path.

    Returns:
        None.
    """
    components = _get_components(ctx)
    service = EvaluationService(components)

    console.print(
        Panel(
            f"[bold]Episodic Memory Pipeline Evaluation[/bold]\n\n"
            f"Scenario: {scenario}\n"
            f"Precision@K: {k}\n"
            f"Save: {'Yes' if save else 'No'}",
            title="Evaluation",
        )
    )

    out_dir = Path(out) if out else None

    with console.status(f"Running {scenario} scenario..."):
        result = service.run_evaluation(
            scenario=scenario, precision_k=k, save=save, out_dir=out_dir
        )

    if not result.success:
        console.print(f"[red]Evaluation failed: {result.error}[/red]")
        return

    console.print(f"\n[green]✓ Evaluation completed in {result.duration_seconds:.2f}s[/green]")

    if result.git_commit:
        console.print(f"[dim]Git commit: {result.git_commit}[/dim]")

    # Show warnings
    for warning in result.warnings:
        console.print(f"[yellow]⚠ {warning}[/yellow]")

    console.print()

    # Render metrics
    render_eval_metrics(result.metrics, verbose=verbose)

    # Show save location
    if save:
        console.print(f"\n[dim]Run saved to: runs/eval/{result.run_id}/[/dim]")
        console.print(
            f"[dim]Compare with: episodic-memory eval-compare {result.run_id} <other_run>[/dim]"
        )


@click.command("eval-compare")
@click.argument("run_a")
@click.argument("run_b")
@click.pass_context
def eval_compare(ctx: click.Context, run_a: str, run_b: str) -> None:
    """Compare two evaluation runs.

    Args:
        ctx: Click context with pipeline initialization settings.
        run_a: Run ID for the first evaluation run.
        run_b: Run ID for the second evaluation run.

    Returns:
        None.
    """
    components = _get_components(ctx)
    service = EvaluationService(components)

    comparison = service.compare_runs(run_a, run_b)

    if comparison is None:
        console.print(f"[red]Could not load one or both runs: {run_a}, {run_b}[/red]")
        console.print("[dim]Available runs:[/dim]")
        for run_id in service.list_runs()[:10]:
            console.print(f"  {run_id}")
        return

    console.print(
        Panel(
            f"[bold]Evaluation Comparison[/bold]\n\n"
            f"Run A: {comparison.run_a_id}\n"
            f"Run B: {comparison.run_b_id}",
            title="Compare",
        )
    )

    render_eval_comparison(comparison)


@click.command("eval-list")
@click.option("--limit", "-n", default=10, help="Maximum number of runs to show")
@click.pass_context
def eval_list(ctx: click.Context, limit: int) -> None:
    """List available evaluation runs.

    Args:
        ctx: Click context with pipeline initialization settings.
        limit: Maximum number of runs to display.

    Returns:
        None.
    """
    components = _get_components(ctx)
    service = EvaluationService(components)

    runs = service.list_runs()

    if not runs:
        console.print("[yellow]No evaluation runs found.[/yellow]")
        console.print("[dim]Run an evaluation with: episodic-memory eval-run[/dim]")
        return

    table = Table(title=f"Evaluation Runs (showing {min(limit, len(runs))} of {len(runs)})")
    table.add_column("Run ID")
    table.add_column("Timestamp")
    table.add_column("Scenario")
    table.add_column("Success")

    for run_id in runs[:limit]:
        result = service.load_run(run_id)
        if result:
            status = "[green]✓[/green]" if result.success else "[red]✗[/red]"
            table.add_row(
                run_id, result.timestamp[:19] if result.timestamp else "N/A", result.dataset, status
            )

    console.print(table)
    console.print("\n[dim]View details: episodic-memory eval-compare <run_a> <run_b>[/dim]")
