"""Doctor command for the episodic memory CLI."""
import click
from rich.panel import Panel
from rich.table import Table
from typing import TYPE_CHECKING

from ..render import console
from src.services import DiagnosticsService

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
    return get_pipeline_components(ctx.obj.get('use_mock', False))


@click.command()
@click.option('--dry', is_flag=True, help='Dry-run mode: inspect config only, no initialization')
@click.pass_context
def doctor(ctx: click.Context, dry: bool) -> None:
    """
    Run system diagnostics and show configuration status.
    
    This command inspects configuration, provider selection, and bootstrap state
    without making any LLM calls or modifying data.

    Args:
        ctx: Click context with pipeline initialization settings.
        dry: If True, only inspect configuration without initializing components.

    Returns:
        None.
    """
    use_mock = ctx.obj.get('use_mock', False)
    
    if dry:
        _doctor_dry_run(use_mock)
    else:
        _doctor_full(ctx, use_mock)


def _doctor_dry_run(use_mock: bool) -> None:
    """Run doctor in dry-run mode (no component initialization).

    Args:
        use_mock: If True, force mock-provider predictions.

    Returns:
        None.
    """
    service = DiagnosticsService()
    result = service.run_dry_diagnostics(force_mock=use_mock)
    
    console.print(Panel(
        "[bold]Episodic Memory Pipeline - System Diagnostics[/bold]\n"
        "[yellow]DRY RUN — no components initialized[/yellow]",
        title="Doctor (Dry Run)",
        border_style="yellow"
    ))
    
    # Environment Variables
    table1 = Table(title="Environment Variables", show_header=True, header_style="bold cyan")
    table1.add_column("Variable", style="dim")
    table1.add_column("Value")
    table1.add_column("Effect")
    
    env = result["env_vars"]
    resolved = result["resolved"]
    
    for var in ["EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBEDDING_DEVICE", 
                "EMBEDDING_DIMENSION", "LLM_PROVIDER", "OLLAMA_MODEL", 
                "OPENAI_API_KEY", "TOKENIZERS_PARALLELISM"]:
        value = env.get(var)
        display_val = f"[cyan]{value}[/cyan]" if value else "[dim]not set[/dim]"
        
        # Determine effect
        if var == "EMBEDDING_PROVIDER":
            effect = f"→ {resolved['embedding_provider']}"
        elif var == "EMBEDDING_MODEL":
            effect = f"→ {resolved['embedding_model']}"
        elif var == "EMBEDDING_DEVICE":
            effect = f"→ {resolved['embedding_device']}"
        elif var == "EMBEDDING_DIMENSION":
            effect = f"→ {resolved['embedding_dimension']}"
        elif var == "LLM_PROVIDER":
            effect = f"→ {resolved['llm_provider']}"
        elif var == "OLLAMA_MODEL":
            effect = f"→ {resolved['llm_model']}" if resolved['llm_provider'] == 'ollama' else "[dim]N/A[/dim]"
        elif var == "OPENAI_API_KEY":
            effect = "Required for OpenAI provider"
        elif var == "TOKENIZERS_PARALLELISM":
            effect = "[green]safe[/green]" if value == "false" else "[yellow]should be 'false'[/yellow]"
        else:
            effect = ""
        
        table1.add_row(var, display_val, effect)
    
    console.print(table1)
    console.print()
    
    # Provider Selection Preview
    pred = result["predictions"]
    
    table2 = Table(title="Provider Selection (Predicted)", show_header=True, header_style="bold cyan")
    table2.add_column("Component", style="dim")
    table2.add_column("Will Use")
    table2.add_column("Status")
    
    if use_mock:
        emb_status = "[yellow]⚠ MOCK (--mock flag)[/yellow]"
    elif pred["will_use_mock_embeddings"]:
        emb_status = "[yellow]⚠ MOCK[/yellow]"
    else:
        emb_status = "[green]✓ Real[/green]"
    
    table2.add_row(
        "Embeddings",
        "mock" if pred["will_use_mock_embeddings"] else resolved['embedding_provider'],
        emb_status
    )
    
    if use_mock:
        llm_status = "[yellow]⚠ MOCK (--mock flag)[/yellow]"
    elif pred["will_use_mock_llm"]:
        llm_status = "[yellow]⚠ MOCK (no API key)[/yellow]"
    else:
        llm_status = "[green]✓ Real[/green]"
    
    table2.add_row(
        "LLM",
        "mock" if pred["will_use_mock_llm"] else resolved['llm_provider'],
        llm_status
    )
    
    console.print(table2)
    console.print()
    
    # Suggestions
    if result["suggestions"]:
        suggestion_text = "\n".join(result["suggestions"])
        console.print(Panel(
            f"[bold]Copy-paste these commands to fix issues:[/bold]\n\n"
            f"[cyan]{suggestion_text}[/cyan]",
            title="Suggested Fixes",
            border_style="blue"
        ))
    
    console.print()
    console.print("[dim]Run without --dry to see full diagnostics with initialized components.[/dim]")


def _doctor_full(ctx: click.Context, use_mock: bool) -> None:
    """Run doctor with full component initialization.

    Args:
        ctx: Click context with pipeline initialization settings.
        use_mock: If True, force mock providers for initialized components.

    Returns:
        None.
    """
    components = _get_components(ctx)
    service = DiagnosticsService(components=components)
    
    console.print(Panel(
        "[bold]Episodic Memory Pipeline - System Diagnostics[/bold]",
        title="Doctor",
        border_style="blue"
    ))
    
    result = service.run_full_diagnostics()
    
    # Bootstrap Status
    table1 = Table(title="Bootstrap Status", show_header=True, header_style="bold cyan")
    table1.add_column("Check", style="dim")
    table1.add_column("Status")
    table1.add_column("Details")
    
    bs = result.bootstrap
    table1.add_row(
        "Bootstrap initialized",
        "[green]✓ YES[/green]" if bs.is_initialized else "[dim]NO[/dim]",
        "FAISS/SentenceTransformers init order enforced" if bs.is_initialized else "Not using bootstrap"
    )
    table1.add_row(
        "Embedding model preloaded",
        "[green]✓ YES[/green]" if bs.has_cached_model else "[dim]NO[/dim]",
        "Model cached in memory" if bs.has_cached_model else "No preloaded model"
    )
    table1.add_row(
        "TOKENIZERS_PARALLELISM",
        "[green]✓ YES[/green]" if bs.tokenizers_parallelism_disabled else "[yellow]⚠ NO[/yellow]",
        "false (safe)" if bs.tokenizers_parallelism_disabled else "not set (may cause issues)"
    )
    
    console.print(table1)
    console.print()
    
    # LLM Provider
    table2 = Table(title="LLM Provider", show_header=True, header_style="bold cyan")
    table2.add_column("Property", style="dim")
    table2.add_column("Value")
    
    llm = result.llm
    table2.add_row("Provider type", llm.type)
    table2.add_row("Model", llm.model)
    table2.add_row("Temperature", llm.temperature)
    table2.add_row("Is mock", "[green]YES[/green]" if llm.is_mock else "[dim]NO[/dim]")
    if llm.base_url:
        table2.add_row("Base URL", llm.base_url)
    
    console.print(table2)
    console.print()
    
    # Embedding Provider
    table3 = Table(title="Embedding Provider", show_header=True, header_style="bold cyan")
    table3.add_column("Property", style="dim")
    table3.add_column("Value")
    
    emb = result.embedding
    table3.add_row("Provider type", emb.type)
    table3.add_row("Model", emb.model)
    table3.add_row("Device", emb.device)
    table3.add_row("Dimension", str(emb.dimension))
    table3.add_row("Is mock", "[green]YES[/green]" if emb.is_mock else "[dim]NO[/dim]")
    
    console.print(table3)
    console.print()
    
    # Vector Store
    table4 = Table(title="Vector Store (FAISS)", show_header=True, header_style="bold cyan")
    table4.add_column("Property", style="dim")
    table4.add_column("Value")
    
    vs = result.vector_store
    table4.add_row("Index type", vs.index_type)
    table4.add_row("Similarity metric", vs.similarity_metric)
    table4.add_row("Index dimension", str(vs.dimension))
    
    dim_status = "[green]✓ Match[/green]" if vs.dimension_match else f"[red]✗ MISMATCH[/red]"
    table4.add_row("Dimension consistency", dim_status)
    
    for idx_name, count in vs.indexes.items():
        table4.add_row(f"  {idx_name} vectors", str(count))
    table4.add_row("Total vectors", str(vs.total_vectors))
    
    console.print(table4)
    console.print()
    
    # Evaluation Readiness
    table5 = Table(title="Evaluation Readiness", show_header=True, header_style="bold cyan")
    table5.add_column("Check", style="dim")
    table5.add_column("Status")
    table5.add_column("Impact")
    
    er = result.eval_readiness
    
    if emb.is_mock:
        table5.add_row("Embeddings", "[yellow]⚠ MOCK[/yellow]", "Retrieval metrics will be SKIPPED")
    else:
        table5.add_row("Embeddings", "[green]✓ Real[/green]", f"Using {emb.type} ({emb.model})")
    
    if llm.is_mock:
        table5.add_row("LLM", "[yellow]⚠ MOCK[/yellow]", "Fact/consolidation metrics may not be meaningful")
    else:
        table5.add_row("LLM", "[green]✓ Real[/green]", f"Using {llm.type} ({llm.model})")
    
    if not vs.dimension_match:
        table5.add_row("Dimensions", "[red]✗ MISMATCH[/red]", "Vector store and embedding dimensions don't match!")
    else:
        table5.add_row("Dimensions", "[green]✓ Consistent[/green]", f"All using {emb.dimension}d vectors")
    
    console.print(table5)
    console.print()
    
    # Overall status
    if er.is_ready:
        console.print(Panel(
            "[green]✓ System ready for meaningful evaluation[/green]\n\n"
            "All providers are configured with real models.\n"
            "Run `episodic-memory eval-run` to test.",
            border_style="green"
        ))
    else:
        warning_text = "[yellow]⚠ System has warnings that may affect evaluation:[/yellow]\n\n"
        for w in er.warnings:
            warning_text += f"• {w}\n"
        warning_text += "\n[dim]Use real providers for meaningful evaluation results.[/dim]"
        console.print(Panel(warning_text, border_style="yellow"))
    
    # Suggestions
    if result.suggestions:
        suggestion_text = "\n".join(result.suggestions)
        console.print()
        console.print(Panel(
            f"[bold]Copy-paste these commands to fix issues:[/bold]\n\n"
            f"[cyan]{suggestion_text}[/cyan]",
            title="Suggested Fixes",
            border_style="blue"
        ))

