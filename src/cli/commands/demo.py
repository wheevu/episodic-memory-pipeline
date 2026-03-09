"""Demo commands for the episodic memory CLI."""

from typing import TYPE_CHECKING

import click
from rich.panel import Panel

from src.services import IngestionService, RetrievalService

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
@click.pass_context
def demo(ctx: click.Context) -> None:
    """Run a scripted demo showcasing core pipeline features.

    Args:
        ctx: Click context with pipeline initialization settings.

    Returns:
        None.
    """
    from src.cli.commands.query import stats

    console.print(
        Panel(
            "[bold]Episodic Memory Pipeline Demo[/bold]\n\n"
            "This demo will walk through the core functionality:\n"
            "1. Ingesting memories\n"
            "2. Running consolidation\n"
            "3. Querying memories\n"
            "4. Narrative recall",
            title="Welcome",
        )
    )

    components = _get_components(ctx)

    ingestion_service = IngestionService(
        components, worthiness_threshold=components.config.memory_worthiness_threshold
    )
    retrieval_service = RetrievalService(components)

    # Demo memories
    demo_memories = [
        "I started learning Korean today. My goal is to be conversational by March for my Seoul trip.",
        "I've been practicing Korean for 2 hours. Learned basic greetings: 안녕하세요, 감사합니다.",
        "My friend recommended the Talk To Me In Korean podcast. Going to try it tomorrow.",
        "Had my first conversation in Korean today! Just basic stuff but it felt great.",
        "I prefer visual learning over audio. Going to focus more on writing practice.",
        "Booked my flight to Seoul for March 15th. Excited but nervous about the language barrier.",
    ]

    # Ingest demo memories
    console.print("\n[bold cyan]Step 1: Ingesting demo memories...[/bold cyan]\n")

    for text in demo_memories:
        result = ingestion_service.ingest_text(text, source="demo", force=True)
        status = "[green]✓[/green]" if result.success else "[red]✗[/red]"
        console.print(f"  {status} {text[:50]}...")

    console.print("\n[bold cyan]Step 2: Running consolidation...[/bold cyan]\n")

    results = retrieval_service.consolidate(consolidate_all=True)
    if results:
        for r in results:
            console.print(
                f"  Consolidated '{r.topic}': {r.episodes_processed} episodes → {r.summaries_created} summary, {r.facts_extracted} facts"
            )
    else:
        console.print("  No topics needed consolidation")

    console.print("\n[bold cyan]Step 3: Semantic query...[/bold cyan]\n")

    query_text = "What am I learning right now?"
    console.print(f'  Query: "{query_text}"')
    result = retrieval_service.query(query_text)
    console.print(Panel(result.answer, title="Answer"))

    console.print("\n[bold cyan]Step 4: Narrative recall...[/bold cyan]\n")

    recall_text = "Tell me about my Korean learning journey"
    console.print(f'  Query: "{recall_text}"')
    result = retrieval_service.recall_narrative("korean", is_topic=False)
    console.print(Panel(result.answer, title="Narrative"))

    # Final stats
    console.print("\n[bold cyan]Final Statistics:[/bold cyan]\n")
    ctx.invoke(stats)

    console.print("\n[green]Demo complete![/green]")


@click.command()
@click.pass_context
def interactive(ctx: click.Context) -> None:
    """Start an interactive REPL-like session.

    Args:
        ctx: Click context with pipeline initialization settings.

    Returns:
        None.
    """
    components = _get_components(ctx)

    ingestion_service = IngestionService(
        components, worthiness_threshold=components.config.memory_worthiness_threshold
    )
    retrieval_service = RetrievalService(components)

    console.print(
        Panel(
            "[bold]Interactive Memory Session[/bold]\n\n"
            "Commands:\n"
            "  /remember <text> - Store a memory\n"
            "  /query <text>    - Query memories\n"
            "  /recall <topic>  - Recall narrative\n"
            "  /stats           - Show statistics\n"
            "  /quit            - Exit\n\n"
            "Or just type naturally - it will be analyzed for memory-worthiness.",
            title="Welcome",
        )
    )

    while True:
        try:
            user_input = console.input("\n[bold]>[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.lower() in ["/quit", "/exit", "/q"]:
            console.print("[dim]Goodbye![/dim]")
            break

        if user_input.startswith("/remember "):
            text = user_input[10:]
            result = ingestion_service.ingest_text(text, source="interactive", force=True)
            if result.success:
                console.print(f"[green]✓ Remembered:[/green] {result.episode.content}")
            else:
                console.print(f"[yellow]Not stored:[/yellow] {result.reason}")

        elif user_input.startswith("/query "):
            query_text = user_input[7:]
            result = retrieval_service.query(query_text)
            console.print(Panel(result.answer, title="Answer"))

        elif user_input.startswith("/recall "):
            topic = user_input[8:]
            result = retrieval_service.recall_narrative(topic, is_topic=True)
            console.print(Panel(result.answer, title=f"Narrative: {topic}"))

        elif user_input == "/stats":
            system_stats = retrieval_service.get_stats()
            console.print(
                f"Episodes: {system_stats.total_episodes} | "
                f"Facts: {system_stats.total_facts} | "
                f"Summaries: {system_stats.total_summaries}"
            )

        else:
            # Try to ingest as memory
            result = ingestion_service.ingest_text(user_input, source="interactive")
            if result.success:
                console.print(f"[green]✓ Noted:[/green] {result.episode.content[:50]}...")
            else:
                console.print(f"[dim]({result.reason})[/dim]")
