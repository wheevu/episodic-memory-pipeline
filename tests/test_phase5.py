"""Regression smoke tests kept from phase 5."""


class TestCLIImportSmoke:
    """Verify CLI modules import without SyntaxError."""

    def test_import_src_cli(self) -> None:
        import src.cli  # noqa: F401

    def test_import_src_cli_commands(self) -> None:
        import src.cli.commands  # noqa: F401

    def test_import_ingest_command(self) -> None:
        import src.cli.commands.ingest  # noqa: F401

    def test_cli_group_has_expected_commands(self) -> None:
        from src.cli import cli

        command_names = set(cli.commands.keys()) if hasattr(cli, "commands") else set()
        expected = {"ingest", "query", "stats"}
        missing = expected - command_names
        assert not missing, f"CLI group is missing commands: {missing}"
