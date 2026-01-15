"""Query commands for the episodic memory CLI."""

from typing import TYPE_CHECKING

import click
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from src.services import RetrievalService

from ..render import console

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


@click.command()
@click.argument("query_text")
@click.option("--no-synthesize", is_flag=True, help="Skip answer synthesis")
@click.pass_context
def query(ctx: click.Context, query_text: str, no_synthesize: bool) -> None:
    """Query the memory system.

    Args:
        ctx: Click context with pipeline initialization settings.
        query_text: Natural-language query to execute.
        no_synthesize: If True, skip LLM-based answer synthesis.

    Returns:
        None.
    """
    components = _get_components(ctx)
    service = RetrievalService(components)

    with console.status("Searching memories..."):
        result = service.query(query_text, synthesize=not no_synthesize)

    # Display answer
    if result.answer:
        console.print(
            Panel(
                Markdown(result.answer),
                title=f"Answer (confidence: {result.confidence:.1%})",
                border_style="green" if result.confidence > 0.7 else "yellow",
            )
        )

    # Display supporting evidence
    if result.facts:
        console.print("\n[bold]Related Facts:[/bold]")
        for fact in result.facts[:5]:
            console.print(f"  • {fact.content} [dim](conf: {fact.confidence:.1%})[/dim]")

    if result.episodes:
        console.print(f"\n[bold]Supporting Episodes:[/bold] ({len(result.episodes)} found)")
        for ep in result.episodes[:5]:
            date_str = ep.occurred_at.strftime("%Y-%m-%d")
            console.print(f"  • [{date_str}] {ep.content[:80]}...")

    if result.gaps:
        console.print(f"\n[dim]Gaps: {', '.join(result.gaps)}[/dim]")


@click.command()
@click.argument("topic_or_query")
@click.option("--topic", is_flag=True, help="Treat input as exact topic name")
@click.pass_context
def recall(ctx: click.Context, topic_or_query: str, topic: bool) -> None:
    """Recall the narrative/journey for a topic.

    Args:
        ctx: Click context with pipeline initialization settings.
        topic_or_query: Topic name or free-form query used to infer a topic.
        topic: If True, treat `topic_or_query` as an exact topic name.

    Returns:
        None.
    """
    components = _get_components(ctx)
    service = RetrievalService(components)

    with console.status("Recalling narrative..."):
        result = service.recall_narrative(topic_or_query, is_topic=topic)

    # Display narrative
    console.print(
        Panel(Markdown(result.answer), title=f"Narrative: {topic_or_query}", border_style="blue")
    )

    # Display timeline
    if result.episodes:
        console.print("\n[bold]Timeline:[/bold]")
        for ep in result.episodes[:10]:
            date_str = ep.occurred_at.strftime("%Y-%m-%d %H:%M")
            console.print(f"  [{date_str}] {ep.content[:60]}...")


@click.command()
@click.option("--topic", help="Consolidate specific topic")
@click.option("--all", "consolidate_all", is_flag=True, help="Consolidate all topics needing it")
@click.pass_context
def consolidate(ctx: click.Context, topic: str, consolidate_all: bool) -> None:
    """Run memory consolidation.

    Args:
        ctx: Click context with pipeline initialization settings.
        topic: Optional topic name to consolidate.
        consolidate_all: If True, consolidate all topics meeting criteria.

    Returns:
        None.
    """
    components = _get_components(ctx)
    service = RetrievalService(components)

    if not topic and not consolidate_all:
        console.print("[red]Please specify --topic or --all[/red]")
        return

    with console.status(f"Consolidating{' topic: ' + topic if topic else ' all topics'}..."):
        results = service.consolidate(topic=topic, consolidate_all=consolidate_all)

    if not results:
        console.print("[yellow]No topics needed consolidation[/yellow]")
        return

    # Display results
    table = Table(title="Consolidation Results")
    table.add_column("Topic")
    table.add_column("Episodes")
    table.add_column("Summaries")
    table.add_column("Facts")
    table.add_column("Duration")

    for r in results:
        table.add_row(
            r.topic or "all",
            str(r.episodes_processed),
            str(r.summaries_created),
            str(r.facts_extracted),
            f"{r.duration_seconds:.2f}s",
        )

    console.print(table)


@click.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show memory system statistics.

    Args:
        ctx: Click context with pipeline initialization settings.

    Returns:
        None.
    """
    components = _get_components(ctx)
    service = RetrievalService(components)

    system_stats = service.get_stats()

    # Database stats
    table = Table(title="Database Statistics")
    table.add_column("Metric")
    table.add_column("Value")

    table.add_row("Total Episodes", str(system_stats.total_episodes))
    table.add_row("Unconsolidated Episodes", str(system_stats.unconsolidated_episodes))
    table.add_row("Total Facts", str(system_stats.total_facts))
    table.add_row("Total Summaries", str(system_stats.total_summaries))
    table.add_row("Total Topics", str(system_stats.total_topics))

    console.print(table)

    # Vector stats
    table2 = Table(title="Vector Store Statistics")
    table2.add_column("Index")
    table2.add_column("Count")
    table2.add_column("Dimension")

    for name, info in system_stats.vector_stats.items():
        table2.add_row(name, str(info["count"]), str(info["dimension"]))

    console.print(table2)

    # Topics
    topics = service.get_topics()
    if topics:
        table3 = Table(title="Topics")
        table3.add_column("Name")
        table3.add_column("Episodes")
        table3.add_column("Last Consolidated")

        for t in topics[:10]:
            last_cons = t.get("last_consolidation") or "never"
            table3.add_row(t["name"], str(t["episode_count"]), str(last_cons))

        console.print(table3)
