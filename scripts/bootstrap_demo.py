#!/usr/bin/env python3
"""
Bootstrap script for setting up demo data and FAISS indexes.

This script creates a reproducible demo environment from fixtures.
Run this after a fresh clone to generate local artifacts.

Usage:
    python -m scripts.bootstrap_demo
    python -m scripts.bootstrap_demo --clean     # Clear existing data first
    python -m scripts.bootstrap_demo --mock      # Use mock providers (fast, no models)
    python -m scripts.bootstrap_demo --dry-run   # Show what would be done
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed CLI arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Bootstrap demo data for the Episodic Memory Pipeline"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing data before bootstrapping"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock providers (fast, no model downloads)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=PROJECT_ROOT / "demo_data" / "fixtures.json",
        help="Path to fixtures JSON file"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Directory for generated artifacts"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    return parser.parse_args()


def log(message: str, quiet: bool = False) -> None:
    """Print a message unless quiet mode is enabled.

    Args:
        message: Message to print.
        quiet: If True, suppress output.

    Returns:
        None.
    """
    if not quiet:
        print(message)


def load_fixtures(fixtures_path: Path) -> dict:
    """Load fixtures from a JSON file.

    Args:
        fixtures_path: Path to the fixtures JSON file.

    Returns:
        Parsed fixtures dictionary.

    Raises:
        FileNotFoundError: If `fixtures_path` does not exist.
        json.JSONDecodeError: If the fixtures file is not valid JSON.
    """
    if not fixtures_path.exists():
        raise FileNotFoundError(f"Fixtures file not found: {fixtures_path}")
    
    with open(fixtures_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data


def clean_data_directory(data_dir: Path, dry_run: bool = False, quiet: bool = False) -> None:
    """Remove existing generated artifacts from a data directory.

    Args:
        data_dir: Directory containing generated artifacts.
        dry_run: If True, only report what would be removed.
        quiet: If True, suppress output.

    Returns:
        None.
    """
    patterns = ["*.db", "*.db-shm", "*.db-wal", "*.faiss", "*.npy", "*.index"]
    
    removed = []
    for pattern in patterns:
        for filepath in data_dir.glob(pattern):
            removed.append(filepath.name)
            if not dry_run:
                filepath.unlink()
    
    if removed:
        log(f"  Removed: {', '.join(removed)}", quiet)
    else:
        log("  No existing artifacts to remove", quiet)


def ensure_data_directory(data_dir: Path, dry_run: bool = False) -> bool:
    """Create the data directory if it doesn't exist.

    Args:
        data_dir: Directory to create if missing.
        dry_run: If True, do not create the directory.

    Returns:
        True if the directory would be/was created; otherwise False.
    """
    if not data_dir.exists():
        if not dry_run:
            data_dir.mkdir(parents=True, exist_ok=True)
        return True
    return False


def bootstrap_demo(
    fixtures_path: Path,
    data_dir: Path,
    use_mock: bool = False,
    dry_run: bool = False,
    quiet: bool = False
) -> dict:
    """
    Bootstrap demo data from fixtures.
    
    Args:
        fixtures_path: Path to fixtures JSON file
        data_dir: Directory for generated artifacts
        use_mock: Use mock providers (fast, no models)
        dry_run: Don't actually make changes
        quiet: Suppress output
        
    Returns:
        dict with summary of what was generated
    """
    # Set up environment for bootstrap
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    
    summary = {
        "fixtures_path": str(fixtures_path),
        "data_dir": str(data_dir),
        "use_mock": use_mock,
        "dry_run": dry_run,
        "episodes_ingested": 0,
        "artifacts_created": [],
        "errors": [],
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    log("\n" + "=" * 60, quiet)
    log("Episodic Memory Pipeline - Demo Bootstrap", quiet)
    log("=" * 60, quiet)
    
    # Load fixtures
    log(f"\n[1/4] Loading fixtures from {fixtures_path}...", quiet)
    try:
        fixtures = load_fixtures(fixtures_path)
        episodes = fixtures.get("episodes", [])
        log(f"  Found {len(episodes)} episodes to ingest", quiet)
    except Exception as e:
        summary["errors"].append(f"Failed to load fixtures: {e}")
        log(f"  ERROR: {e}", quiet)
        return summary
    
    if dry_run:
        log("\n[DRY RUN] Would perform the following actions:", quiet)
        log(f"  - Create data directory: {data_dir}", quiet)
        log(f"  - Ingest {len(episodes)} episodes", quiet)
        log(f"  - Generate FAISS indexes", quiet)
        log(f"  - Generate ID map files (.npy)", quiet)
        summary["episodes_ingested"] = len(episodes)
        return summary
    
    # Ensure data directory exists
    log(f"\n[2/4] Setting up data directory: {data_dir}", quiet)
    created = ensure_data_directory(data_dir, dry_run)
    if created:
        log("  Created data directory", quiet)
    else:
        log("  Data directory already exists", quiet)
    
    # Import pipeline components (after setting TOKENIZERS_PARALLELISM)
    log(f"\n[3/4] Initializing pipeline (mock={use_mock})...", quiet)
    
    try:
        from config import Config
        from src.bootstrap import get_components
        
        # Create config with our data directory
        config = Config()
        config.database_path = data_dir / "memory.db"
        config.vector_index_path = data_dir / "vectors.faiss"
        
        # Get components
        components = get_components(config=config, force_mock=use_mock, verbose=not quiet)
        
    except Exception as e:
        summary["errors"].append(f"Failed to initialize pipeline: {e}")
        log(f"  ERROR: {e}", quiet)
        return summary
    
    # Ingest episodes
    log(f"\n[4/4] Ingesting {len(episodes)} episodes...", quiet)
    
    try:
        ingestion = components.IngestionPipeline(
            components.database,
            components.vector_store,
            components.embedding_provider,
            components.llm,
            worthiness_threshold=0.3  # Lower threshold for demo
        )
        
        success_count = 0
        for i, ep in enumerate(episodes):
            text = ep.get("text", "")
            source = ep.get("source", "demo")
            
            result = ingestion.ingest(text, source=source, force=True)
            
            if result.success:
                success_count += 1
                if not quiet:
                    status = "✓"
            else:
                if not quiet:
                    status = "✗"
                summary["errors"].append(f"Episode {i}: {result.reason}")
            
            if not quiet:
                # Truncate long text for display
                display_text = text[:50] + "..." if len(text) > 50 else text
                print(f"  {status} [{i+1}/{len(episodes)}] {display_text}")
        
        summary["episodes_ingested"] = success_count
        
        # Save vector store to persist indexes
        components.vector_store.save()
        
    except Exception as e:
        summary["errors"].append(f"Ingestion error: {e}")
        log(f"  ERROR: {e}", quiet)
        return summary
    
    # List created artifacts
    log("\n" + "-" * 60, quiet)
    log("Generated Artifacts:", quiet)
    
    for filepath in sorted(data_dir.glob("*")):
        if filepath.is_file():
            size_kb = filepath.stat().st_size / 1024
            summary["artifacts_created"].append(filepath.name)
            log(f"  {filepath.name:40} {size_kb:>8.1f} KB", quiet)
    
    # Get stats
    try:
        db_stats = components.database.get_statistics()
        vec_stats = components.vector_store.get_statistics()
        
        log("\n" + "-" * 60, quiet)
        log("Summary:", quiet)
        log(f"  Episodes in DB:     {db_stats['total_episodes']}", quiet)
        log(f"  Facts extracted:    {db_stats['total_facts']}", quiet)
        log(f"  Summaries:          {db_stats['total_summaries']}", quiet)
        log(f"  Topics:             {db_stats['total_topics']}", quiet)
        
        for idx_name, idx_info in vec_stats.items():
            log(f"  Vectors ({idx_name}): {idx_info['count']}", quiet)
            
    except Exception as e:
        log(f"  Could not get stats: {e}", quiet)
    
    log("\n" + "=" * 60, quiet)
    if summary["errors"]:
        log(f"Completed with {len(summary['errors'])} error(s)", quiet)
    else:
        log("Bootstrap complete!", quiet)
    log("=" * 60 + "\n", quiet)
    
    return summary


def main() -> None:
    """Main entry point for demo bootstrapping.

    Returns:
        None.
    """
    args = parse_args()
    
    # Handle clean option
    if args.clean:
        log("\n[0/4] Cleaning existing data...", args.quiet)
        if args.data_dir.exists():
            clean_data_directory(args.data_dir, args.dry_run, args.quiet)
        else:
            log("  Data directory doesn't exist, nothing to clean", args.quiet)
    
    # Run bootstrap
    summary = bootstrap_demo(
        fixtures_path=args.fixtures,
        data_dir=args.data_dir,
        use_mock=args.mock,
        dry_run=args.dry_run,
        quiet=args.quiet
    )
    
    # Exit with error code if there were errors
    if summary["errors"]:
        sys.exit(1)
    
    return summary


if __name__ == "__main__":
    main()

