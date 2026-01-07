"""Ingest command for the episodic memory CLI."""
import click
from rich.panel import Panel
from typing import TYPE_CHECKING

from ..render import console
from src.services import IngestionService

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
@click.argument('text')
@click.option('--source', default='cli', help='Source of the input')
@click.option('--force', is_flag=True, help='Skip worthiness check')
@click.pass_context
def ingest(ctx: click.Context, text: str, source: str, force: bool) -> None:
    """Ingest a piece of text into memory.

    Args:
        ctx: Click context with pipeline initialization settings.
        text: Input text to ingest.
        source: Source label associated with the input.
        force: If True, bypass worthiness checks and store anyway.

    Returns:
        None.
    """
    from config import config
    
    components = _get_components(ctx)
    service = IngestionService(
        components, 
        worthiness_threshold=config.memory_worthiness_threshold
    )
    
    with console.status("Processing input..."):
        result = service.ingest_text(text, source=source, force=force)
    
    if result.success:
        ep = result.episode
        console.print(Panel(
            f"[green]✓ Memory stored successfully[/green]\n\n"
            f"[bold]Content:[/bold] {ep.content}\n"
            f"[bold]Type:[/bold] {ep.memory_type}\n"
            f"[bold]Topics:[/bold] {', '.join(ep.topics) or 'none'}\n"
            f"[bold]Importance:[/bold] {ep.importance:.2f}\n"
            f"[bold]ID:[/bold] {ep.id[:8]}",
            title="Ingestion Result"
        ))
    else:
        console.print(Panel(
            f"[yellow]⊘ Not stored[/yellow]\n\n"
            f"[bold]Reason:[/bold] {result.reason}\n"
            f"[bold]Confidence:[/bold] {result.classification_confidence:.2f if result.classification_confidence else 'N/A'}",
            title="Ingestion Result"
        ))

