"""Interactive D3 Renderer for knowledge graph visualization.

Generates HTML files with embedded D3.js for interactive
graph exploration with zoom, pan, and filter support."""

from __future__ import annotations

import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ingestforge.viz.graph_export import GraphData, GraphExporter
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_HTML_SIZE_MB = 10
D3_CDN_URL = "https://d3js.org/d3.v7.min.js"

# Template directory
TEMPLATES_DIR = Path(__file__).parent / "templates"


@dataclass
class RenderConfig:
    """Configuration for D3 rendering."""

    width: int = 960
    height: int = 600
    node_radius: int = 8
    link_distance: int = 100
    charge_strength: int = -300
    show_labels: bool = True
    show_legend: bool = True
    enable_zoom: bool = True
    enable_drag: bool = True


class D3Renderer:
    """Renders knowledge graphs as interactive D3 visualizations.

    Generates standalone HTML files that can be opened
    in any modern web browser.
    """

    def __init__(
        self,
        config: Optional[RenderConfig] = None,
    ) -> None:
        """Initialize the renderer.

        Args:
            config: Render configuration
        """
        self.config = config or RenderConfig()
        self._exporter = GraphExporter()

    def render(
        self,
        graph_data: GraphData,
        output_path: Path,
    ) -> bool:
        """Render graph to HTML file.

        Args:
            graph_data: Graph data to render
            output_path: Output HTML file path

        Returns:
            True if successful
        """
        if not graph_data.nodes:
            logger.warning("Empty graph, nothing to render")
            return False

        try:
            html_content = self._generate_html(graph_data)
            output_path.write_text(html_content, encoding="utf-8")
            logger.info(f"Graph rendered to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to render graph: {e}")
            return False

    def _generate_html(self, graph_data: GraphData) -> str:
        """Generate complete HTML document.

        Args:
            graph_data: Graph data

        Returns:
            HTML string
        """
        # Get JSON data
        json_data = self._exporter.to_json(graph_data)

        # Generate HTML components
        head = self._generate_head(graph_data.title)
        styles = self._generate_styles()
        body = self._generate_body(graph_data, json_data)
        script = self._generate_script()

        return f"""<!DOCTYPE html>
<html lang="en">
{head}
<body>
{styles}
{body}
{script}
</body>
</html>"""

    def _generate_head(self, title: str) -> str:
        """Generate HTML head section.

        Args:
            title: Page title

        Returns:
            Head HTML
        """
        return f"""<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="{D3_CDN_URL}"></script>
</head>"""

    def _generate_styles(self) -> str:
        """Generate CSS styles.

        Returns:
            Style HTML
        """
        return f"""<style>
{self._get_base_styles()}
{self._get_node_link_styles()}
{self._get_overlay_styles()}
</style>"""

    def _get_base_styles(self) -> str:
        """Get base body and container styles."""
        return """    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        margin: 0;
        padding: 0;
        background: #f5f5f5;
    }
    #graph-container {
        width: 100%;
        height: 100vh;
        background: white;
        border: 1px solid #ddd;
    }"""

    def _get_node_link_styles(self) -> str:
        """Get styles for graph nodes and links."""
        return """    .node { cursor: pointer; }
    .node circle { stroke: #fff; stroke-width: 2px; }
    .node text { font-size: 12px; fill: #333; pointer-events: none; }
    .link { stroke: #999; stroke-opacity: 0.6; }
    .link.highlighted { stroke: #ff6b6b; stroke-width: 3px; }
    .tooltip {
        position: absolute;
        padding: 8px 12px;
        background: rgba(0, 0, 0, 0.8);
        color: white;
        border-radius: 4px;
        font-size: 12px;
        pointer-events: none;
        z-index: 1000;
    }"""

    def _get_overlay_styles(self) -> str:
        """Get styles for UI overlays (legend, controls)."""
        return """    #legend, #controls, #info {
        position: absolute;
        background: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    #legend { top: 20px; right: 20px; }
    #legend h3 { margin: 0 0 10px 0; font-size: 14px; }
    .legend-item { display: flex; align-items: center; margin: 5px 0; font-size: 12px; }
    .legend-color { width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
    #controls { top: 20px; left: 20px; }
    #controls button {
        display: block; width: 100%; padding: 8px 12px; margin: 5px 0;
        border: none; background: #4e79a7; color: white; border-radius: 4px; cursor: pointer;
    }
    #controls button:hover { background: #3d5f8a; }
    #info { bottom: 20px; left: 20px; font-size: 12px; }"""

    def _generate_body(self, graph_data: GraphData, json_data: str) -> str:
        """Generate HTML body.

        Args:
            graph_data: Graph data
            json_data: JSON string

        Returns:
            Body HTML
        """
        legend = self._generate_legend() if self.config.show_legend else ""
        controls = self._generate_controls()
        info = self._generate_info(graph_data)

        return f"""<div id="graph-container"></div>
{controls}
{legend}
{info}
<div id="tooltip" class="tooltip" style="display: none;"></div>
<script>
    const graphData = {json_data};
</script>"""

    def _generate_legend(self) -> str:
        """Generate legend HTML.

        Returns:
            Legend HTML
        """
        return """<div id="legend">
    <h3>Node Types</h3>
    <div class="legend-item">
        <div class="legend-color" style="background: #4e79a7;"></div>
        <span>Document</span>
    </div>
    <div class="legend-item">
        <div class="legend-color" style="background: #f28e2c;"></div>
        <span>Concept</span>
    </div>
    <div class="legend-item">
        <div class="legend-color" style="background: #e15759;"></div>
        <span>Entity</span>
    </div>
    <div class="legend-item">
        <div class="legend-color" style="background: #76b7b2;"></div>
        <span>Topic</span>
    </div>
</div>"""

    def _generate_controls(self) -> str:
        """Generate controls HTML.

        Returns:
            Controls HTML
        """
        return """<div id="controls">
    <button onclick="resetZoom()">Reset View</button>
    <button onclick="toggleLabels()">Toggle Labels</button>
</div>"""

    def _generate_info(self, graph_data: GraphData) -> str:
        """Generate info panel HTML.

        Args:
            graph_data: Graph data

        Returns:
            Info HTML
        """
        return f"""<div id="info">
    <strong>{graph_data.title}</strong><br>
    Nodes: {graph_data.node_count} | Edges: {graph_data.edge_count}
</div>"""

    def _generate_script(self) -> str:
        """Generate D3 visualization script.

        Returns:
            Script HTML
        """
        return f"""<script>
{self._get_script_setup()}
{self._get_script_simulation()}
{self._get_script_rendering()}
{self._get_script_events()}
{self._get_script_controls()}
</script>"""

    def _get_script_setup(self) -> str:
        """Get script initialization logic."""
        cfg = self.config
        return f"""const width = {cfg.width};
const height = {cfg.height};
let labelsVisible = {str(cfg.show_labels).lower()};

const svg = d3.select("#graph-container")
    .append("svg")
    .attr("width", "100%").attr("height", "100%")
    .attr("viewBox", [0, 0, width, height]);

const g = svg.append("g");
const zoom = d3.zoom().scaleExtent([0.1, 4])
    .on("zoom", (event) => g.attr("transform", event.transform));
svg.call(zoom);

const color = d3.scaleOrdinal()
    .domain(["document", "concept", "entity", "topic", "chunk"])
    .range(["#4e79a7", "#f28e2c", "#e15759", "#76b7b2", "#59a14f"]);"""

    def _get_script_simulation(self) -> str:
        """Get force simulation logic."""
        cfg = self.config
        return f"""const simulation = d3.forceSimulation(graphData.nodes)
    .force("link", d3.forceLink(graphData.links).id(d => d.id).distance({cfg.link_distance}))
    .force("charge", d3.forceManyBody().strength({cfg.charge_strength}))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide().radius({cfg.node_radius + 5}));"""

    def _get_script_rendering(self) -> str:
        """Get node and link rendering logic."""
        cfg = self.config
        return f"""const link = g.append("g").selectAll("line")
    .data(graphData.links).join("line")
    .attr("class", "link").attr("stroke-width", d => Math.sqrt(d.weight || 1));

const node = g.append("g").selectAll("g")
    .data(graphData.nodes).join("g")
    .attr("class", "node")
    .call(d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended));

node.append("circle")
    .attr("r", d => (d.size || 1) * {cfg.node_radius})
    .attr("fill", d => d.color || color(d.type));

node.append("text").attr("x", 12).attr("dy", ".35em")
    .text(d => d.label).style("display", labelsVisible ? "block" : "none");"""

    def _get_script_events(self) -> str:
        """Get event handling logic (hover, simulation tick, drag)."""
        return """const tooltip = d3.select("#tooltip");
node.on("mouseover", (event, d) => {
    tooltip.style("display", "block").style("left", (event.pageX + 10) + "px")
        .style("top", (event.pageY - 10) + "px")
        .html(`<strong>${d.label}</strong><br>Type: ${d.type}`);
    link.classed("highlighted", l => l.source.id === d.id || l.target.id === d.id);
}).on("mouseout", () => {
    tooltip.style("display", "none"); link.classed("highlighted", false);
});

simulation.on("tick", () => {
    link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    node.attr("transform", d => `translate(${d.x},${d.y})`);
});

function dragstarted(event) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    event.subject.fx = event.subject.x; event.subject.fy = event.subject.y;
}
function dragged(event) { event.subject.fx = event.x; event.subject.fy = event.y; }
function dragended(event) {
    if (!event.active) simulation.alphaTarget(0);
    event.subject.fx = null; event.subject.fy = null;
}"""

    def _get_script_controls(self) -> str:
        """Get UI control functions."""
        return """function resetZoom() {
    svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
}
function toggleLabels() {
    labelsVisible = !labelsVisible;
    node.selectAll("text").style("display", labelsVisible ? "block" : "none");
}"""


def create_renderer(
    width: int = 960,
    height: int = 600,
) -> D3Renderer:
    """Factory function to create renderer.

    Args:
        width: Visualization width
        height: Visualization height

    Returns:
        Configured D3Renderer
    """
    config = RenderConfig(width=width, height=height)
    return D3Renderer(config=config)


def render_graph(
    graph_data: GraphData,
    output_path: Path,
) -> bool:
    """Convenience function to render graph.

    Args:
        graph_data: Graph data to render
        output_path: Output HTML path

    Returns:
        True if successful
    """
    renderer = create_renderer()
    return renderer.render(graph_data, output_path)


def render_from_networkx(
    graph: object,
    output_path: Path,
) -> bool:
    """Render NetworkX graph to HTML.

    Args:
        graph: NetworkX graph
        output_path: Output HTML path

    Returns:
        True if successful
    """
    exporter = GraphExporter()
    graph_data = exporter.from_networkx(graph)
    return render_graph(graph_data, output_path)


def render_with_template(
    graph_data: GraphData,
    output_path: Path,
    title: Optional[str] = None,
    api_endpoint: Optional[str] = None,
    library_filter: Optional[str] = None,
) -> bool:
    """Render graph using the enhanced template.

    Uses the templates/graph.html template for dark-themed,
    feature-rich visualization with search, zoom, and filtering.

    Args:
        graph_data: Graph data to render
        output_path: Output HTML path
        title: Optional custom title
        api_endpoint: Optional API endpoint for dynamic loading
        library_filter: Optional library filter for API

    Returns:
        True if successful
    """
    if not graph_data.nodes and not api_endpoint:
        logger.warning("Empty graph and no API endpoint")
        return False

    try:
        template_path = TEMPLATES_DIR / "graph.html"
        if not template_path.exists():
            logger.error(f"Template not found: {template_path}")
            return False

        template_content = template_path.read_text(encoding="utf-8")

        # Convert graph data to JSON
        exporter = GraphExporter()
        json_data = exporter.to_json(graph_data)

        # Render template with data
        html_content = _render_template(
            template_content,
            graph_data=json_data,
            title=title or graph_data.title,
            api_endpoint=api_endpoint or "",
            library_filter=library_filter or "",
        )

        output_path.write_text(html_content, encoding="utf-8")
        logger.info(f"Graph rendered to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to render graph: {e}")
        return False


def _render_template(
    template: str,
    graph_data: str,
    title: str,
    api_endpoint: str,
    library_filter: str,
) -> str:
    """Render template with variables.

    Simple template rendering without Jinja2 dependency.

    Args:
        template: Template HTML
        graph_data: JSON graph data
        title: Page title
        api_endpoint: API endpoint URL
        library_filter: Library filter value

    Returns:
        Rendered HTML
    """
    # Replace template variables
    # Uses simple pattern: {{ variable | default("value") }}
    result = template

    # Replace graph_data - handle default
    if "{{ graph_data |" in result:
        result = result.replace(
            '{{ graph_data | default(\'{"nodes": [], "links": []}\') | safe }}',
            graph_data,
        )
    else:
        result = result.replace("{{ graph_data }}", graph_data)

    # Replace title
    if "{{ title |" in result:
        result = result.replace(
            '{{ title | default("Knowledge Graph - IngestForge") }}',
            title,
        )
    else:
        result = result.replace("{{ title }}", title)

    # Replace api_endpoint
    if "{{ api_endpoint |" in result:
        result = result.replace(
            "{{ api_endpoint | default('') }}",
            api_endpoint,
        )
    else:
        result = result.replace("{{ api_endpoint }}", api_endpoint)

    # Replace library_filter
    if "{{ library_filter |" in result:
        result = result.replace(
            "{{ library_filter | default('') }}",
            library_filter,
        )
    else:
        result = result.replace("{{ library_filter }}", library_filter)

    return result


def open_in_browser(html_path: Path) -> bool:
    """Open HTML file in default browser.

    Args:
        html_path: Path to HTML file

    Returns:
        True if browser opened
    """
    try:
        if not html_path.exists():
            logger.error(f"File not found: {html_path}")
            return False

        # Convert to file URL
        file_url = html_path.resolve().as_uri()
        webbrowser.open(file_url)
        logger.info(f"Opened in browser: {file_url}")
        return True

    except Exception as e:
        logger.error(f"Failed to open browser: {e}")
        return False
