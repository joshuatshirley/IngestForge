"""Tests for Knowledge Graph Export Command.

Test Strategy
-------------
- Test graph building from chunk metadata
- Test node and edge creation
- Test format validation (json/html)
"""

from pathlib import Path
from unittest.mock import Mock, patch
import pytest
from ingestforge.cli.export.knowledge_graph import KnowledgeGraphExportCommand


class TestKnowledgeGraphExportInit:
    """Tests for initialization."""

    def test_create_command(self):
        """Test creating command instance."""
        cmd = KnowledgeGraphExportCommand()
        assert cmd is not None

    def test_inherits_from_export_command(self):
        """Test inheritance from ExportCommand."""
        from ingestforge.cli.export.base import ExportCommand

        cmd = KnowledgeGraphExportCommand()
        assert isinstance(cmd, ExportCommand)


class TestValidation:
    """Tests for parameter validation."""

    def test_validate_format_json(self):
        """Test JSON format is valid."""
        cmd = KnowledgeGraphExportCommand()
        output = Path("output.json")
        cmd._validate_parameters(output, "json", 100)

    def test_validate_format_html(self):
        """Test HTML format is valid."""
        cmd = KnowledgeGraphExportCommand()
        output = Path("output.html")
        cmd._validate_parameters(output, "html", 100)

    def test_validate_format_invalid(self):
        """Test invalid format raises error."""
        import typer

        cmd = KnowledgeGraphExportCommand()
        output = Path("output.txt")
        with pytest.raises(typer.BadParameter):
            cmd._validate_parameters(output, "invalid", 100)

    def test_validate_max_nodes_bounds(self):
        """Test max_nodes boundary validation."""
        import typer

        cmd = KnowledgeGraphExportCommand()
        output = Path("output.json")

        with pytest.raises(typer.BadParameter):
            cmd._validate_parameters(output, "json", 0)

        with pytest.raises(typer.BadParameter):
            cmd._validate_parameters(output, "json", 501)

        cmd._validate_parameters(output, "json", 1)
        cmd._validate_parameters(output, "json", 500)


class TestGraphBuilding:
    """Tests for graph construction."""

    def test_build_empty_graph(self):
        """Test building graph from empty chunks."""
        cmd = KnowledgeGraphExportCommand()
        graph_data = cmd._build_graph_from_chunks([], 100, True)
        assert graph_data.node_count == 0
        assert graph_data.edge_count == 0

    @patch.object(KnowledgeGraphExportCommand, "extract_chunk_metadata")
    def test_build_graph_with_entities(self, mock_metadata):
        """Test building graph with entities."""
        cmd = KnowledgeGraphExportCommand()
        chunk = Mock()
        chunk.chunk_id = "chunk_1"
        mock_metadata.return_value = {
            "source": "test.txt",
            "entities": ["Entity1", "Entity2"],
            "concepts": [],
        }
        graph_data = cmd._build_graph_from_chunks([chunk], 100, True)
        assert graph_data.node_count == 3
        assert graph_data.edge_count == 2


class TestNodeCreation:
    """Tests for node creation."""

    def test_add_document_node(self):
        """Test adding document node."""
        from ingestforge.viz.graph_export import GraphData, NodeType

        cmd = KnowledgeGraphExportCommand()
        graph_data = GraphData()
        node_ids = set()
        document_nodes = {}

        cmd._add_document_node("test.txt", graph_data, node_ids, document_nodes)

        assert graph_data.node_count == 1
        assert "doc_test.txt" in node_ids
        assert graph_data.nodes[0].node_type == NodeType.DOCUMENT


class TestExport:
    """Tests for export functionality."""

    @patch("ingestforge.cli.export.knowledge_graph.GraphExporter")
    def test_export_json(self, mock_exporter_class):
        """Test exporting as JSON."""
        from ingestforge.viz.graph_export import GraphData

        cmd = KnowledgeGraphExportCommand()
        graph_data = GraphData()
        output = Path("output.json")

        mock_exporter = Mock()
        mock_exporter.to_file.return_value = True
        mock_exporter_class.return_value = mock_exporter

        success = cmd._export_json(graph_data, output)
        assert success is True
        mock_exporter.to_file.assert_called_once_with(graph_data, output)

    @patch("ingestforge.cli.export.knowledge_graph.D3Renderer")
    def test_export_html(self, mock_renderer_class):
        """Test exporting as HTML."""
        from ingestforge.viz.graph_export import GraphData

        cmd = KnowledgeGraphExportCommand()
        graph_data = GraphData()
        output = Path("output.html")

        mock_renderer = Mock()
        mock_renderer.render.return_value = True
        mock_renderer_class.return_value = mock_renderer

        success = cmd._export_html(graph_data, output)
        assert success is True
        mock_renderer.render.assert_called_once_with(graph_data, output)
