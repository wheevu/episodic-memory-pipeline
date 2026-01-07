"""
CLI module for the episodic memory pipeline.

This module provides the Typer/Click-based command-line interface.
All Rich/Typer imports are contained here - services are UI-agnostic.

Usage:
    episodic-memory ingest "I started learning Korean"
    episodic-memory query "What am I learning?"
    episodic-memory doctor
"""
import click
from typing import Optional

from config import config
from src.bootstrap import get_components, PipelineComponents

# Import command modules
from .commands import ingest, query, eval_cmd, doctor, demo

# Global cache for components
_components: Optional[PipelineComponents] = None


def get_pipeline_components(use_mock: bool = False) -> PipelineComponents:
    """
    Get or create pipeline components via bootstrap.
    
    Args:
        use_mock: Force mock providers for testing
        
    Returns:
        PipelineComponents with all initialized components
    """
    global _components
    
    if _components is None or use_mock:
        _components = get_components(config=config, force_mock=use_mock, verbose=True)
    
    return _components


@click.group()
@click.option('--mock', is_flag=True, help='Use mock providers (no API calls)')
@click.pass_context
def app(ctx: click.Context, mock: bool) -> None:
    """Episodic Memory Pipeline CLI entrypoint.

    Args:
        ctx: Click context used to store command-scoped settings.
        mock: If True, force mock providers (no external calls).

    Returns:
        None.
    """
    ctx.ensure_object(dict)
    ctx.obj['use_mock'] = mock


# Register commands
app.add_command(ingest.ingest)
app.add_command(query.query)
app.add_command(query.recall)
app.add_command(query.consolidate)
app.add_command(query.stats)
app.add_command(eval_cmd.eval_cmd, name='eval')
app.add_command(eval_cmd.eval_run)
app.add_command(eval_cmd.eval_compare)
app.add_command(eval_cmd.eval_list)
app.add_command(doctor.doctor)
app.add_command(demo.demo)
app.add_command(demo.interactive)


# For backwards compatibility and direct execution
cli = app

if __name__ == "__main__":
    app()

